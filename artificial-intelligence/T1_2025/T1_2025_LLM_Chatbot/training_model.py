import os
import json
import torch
from huggingface_hub import login
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer
)
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

########################
# CONFIG
########################
HF_TOKEN = "TOKEN"
MODEL_NAME = "llama-2-13b-chat"
REPO_ID = f"meta-llama/{MODEL_NAME}-hf"
CACHE_DIR = "huggingface_cache"
DATASET_PATH = "datasets/no_gpu_limit_500.json" # colab_gpu_limited_100.json # no_gpu_limit_500
N_SAMPLES = 500 #! change to 100 for colab_gpu_limited_100.json
OUTPUT_DIR = f"trained_models/{MODEL_NAME}-lora-output-{N_SAMPLES}"
FINAL_DIR = f"trained_models/{MODEL_NAME}-lora-final-{N_SAMPLES}"

# Quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

# LoRA config
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

########################
# DEVICE MANAGEMENT
########################
def set_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("▶ Using GPU:", torch.cuda.get_device_name(0))
        torch.cuda.empty_cache()
        return device
    else:
        raise RuntimeError("GPU not available, but required for this setup.")

########################
# DATA LOADING
########################
def load_data(file_path, n_samples=N_SAMPLES):
    with open(file_path, 'r') as f:
        data = json.load(f)
    dataset = {
        "question": [item["question"] for item in data],
        "answer": [item["answer"] for item in data]
    }
    ds = Dataset.from_dict(dataset)
    if len(ds) < n_samples:
        print(f"▶ Warning: Dataset has {len(ds)} samples, less than requested {n_samples}")
        n_samples = len(ds)
    return ds.shuffle(seed=42).select(range(n_samples))

########################
# TRAINING FUNCTION
########################
def train_model(data_path=DATASET_PATH, n_samples=N_SAMPLES):
    device = set_device()
    login(token=HF_TOKEN)

    print(f"▶ Loading JSON dataset from {data_path}...")
    train_ds = load_data(data_path, n_samples=n_samples)
    print(f"▶ Using {len(train_ds)} examples for training")

    expected_columns = {"question", "answer"}
    if not all(col in train_ds.column_names for col in expected_columns):
        raise ValueError(f"Dataset must contain {expected_columns}, but found {train_ds.column_names}")

    tokenizer = AutoTokenizer.from_pretrained(REPO_ID, token=HF_TOKEN, cache_dir=CACHE_DIR)
    tokenizer.pad_token = tokenizer.eos_token

    print(f"▶ Loading {MODEL_NAME} in 4-bit...")
    model = AutoModelForCausalLM.from_pretrained(
        REPO_ID,
        quantization_config=bnb_config,
        device_map="auto",
        token=HF_TOKEN,
        cache_dir=CACHE_DIR,
        low_cpu_mem_usage=True
    )
    model = prepare_model_for_kbit_training(model)

    print("▶ Applying LoRA adapters...")
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    def tokenize_function(examples):
        # Enhanced prompt engineering for better fine-tuning
        texts = [f"[INST] {q} [/INST] {a}" for q, a in zip(examples["question"], examples["answer"])]
        tokenized = tokenizer(texts, truncation=True, max_length=512, padding="max_length")
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    print("▶ Tokenizing dataset...")
    tokenized_train = train_ds.map(tokenize_function, batched=True, remove_columns=train_ds.column_names)

    # Updated training arguments for hyperparameter tuning
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        overwrite_output_dir=True,
        num_train_epochs=3,  # Increased epochs for better tuning
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8, #! chaneg to "16" to reduce memory usage
        fp16=True,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=1,
        learning_rate=2e-5,  # Adjusted learning rate
        warmup_steps=50,     # Added warmup for stability
        weight_decay=0.01    # Added regularization
    )

    print("▶ Initializing Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        tokenizer=tokenizer
        # processing_class=tokenizer  # Replace tokenizer=tokenizer #! do it later
    )

    print("▶ Starting training with hyperparameter tuning...")
    trainer.train()

    print(f"▶ Saving LoRA model to {FINAL_DIR}...")
    model.save_pretrained(FINAL_DIR)
    tokenizer.save_pretrained(FINAL_DIR)

    print(f"▶ Merging LoRA weights...")
    merged_model = model.merge_and_unload()
    merged_dir = FINAL_DIR + "_merged"
    merged_model.save_pretrained(merged_dir)
    tokenizer.save_pretrained(merged_dir)

    print(f"▶ Training complete. Model saved in {FINAL_DIR}, merged in {merged_dir}")
    print("▶ Next steps: Evaluate using ROUGE-L, ROUGE-1, ROUGE-2, BERTScore, and BLEU metrics")

if __name__ == "__main__":
    train_model(data_path=DATASET_PATH, n_samples=N_SAMPLES)