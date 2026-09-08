import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset
import evaluate  # Hugging Face's evaluate library
from bert_score import score as bert_score
import pandas as pd
from datetime import datetime

########################
# CONFIG
########################
HF_TOKEN = "YOUR_TOKEN_HERE"  # Replace with your Hugging Face token
MODEL_NAME = "llama-2-7b-chat"
MERGED_MODEL_DIR = f"{MODEL_NAME}-lora-final_merged" #!!! Change to your merged model path
CACHE_DIR = "huggingface_cache"
TEST_DATASET_PATH = "datasets/test_data.json"  # Replace with your test dataset path
REPORT_OUTPUT = f"evaluation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

# Evaluation settings
MAX_LENGTH = 512
BATCH_SIZE = 1

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
        print("▶ No GPU available, using CPU.")
        return torch.device("cpu")

########################
# DATA LOADING
########################
def load_test_data(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    dataset = {
        "question": [item["question"] for item in data],
        "answer": [item["answer"] for item in data]
    }
    return Dataset.from_dict(dataset)

########################
# EVALUATION FUNCTION
########################
def evaluate_model(test_data_path=TEST_DATASET_PATH):
    device = set_device()

    # Load the merged fine-tuned model and tokenizer
    print(f"▶ Loading fine-tuned model from {MERGED_MODEL_DIR}...")
    tokenizer = AutoTokenizer.from_pretrained(MERGED_MODEL_DIR, cache_dir=CACHE_DIR)
    model = AutoModelForCausalLM.from_pretrained(
        MERGED_MODEL_DIR,
        device_map="auto",
        cache_dir=CACHE_DIR,
        low_cpu_mem_usage=True
    ).to(device)
    model.eval()

    # Load test dataset
    print(f"▶ Loading test dataset from {test_data_path}...")
    test_ds = load_test_data(test_data_path)
    print(f"▶ Evaluating on {len(test_ds)} examples")

    # Prepare metrics
    rouge = evaluate.load("rouge")
    bleu = evaluate.load("bleu")
    predictions = []
    references = []

    # Generate predictions
    print("▶ Generating model predictions...")
    for example in test_ds:
        input_text = f"[INST] {example['question']} [/INST]"
        inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=MAX_LENGTH).to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=MAX_LENGTH,
                num_return_sequences=1,
                do_sample=False
            )
        pred = tokenizer.decode(outputs[0], skip_special_tokens=True).replace(input_text, "").strip()
        predictions.append(pred)
        references.append(example["answer"])

    # Compute ROUGE scores
    print("▶ Computing ROUGE scores...")
    rouge_scores = rouge.compute(predictions=predictions, references=references)

    # Compute BLEU score
    print("▶ Computing BLEU score...")
    bleu_score = bleu.compute(predictions=predictions, references=references)

    # Compute BERTScore
    print("▶ Computing BERTScore...")
    P, R, F1 = bert_score(predictions, references, lang="en", verbose=True)
    bertscore_results = {
        "precision": P.mean().item(),
        "recall": R.mean().item(),
        "f1": F1.mean().item()
    }

    # Compile results
    results = {
        "ROUGE-1": rouge_scores["rouge1"],
        "ROUGE-2": rouge_scores["rouge2"],
        "ROUGE-L": rouge_scores["rougeL"],
        "BLEU": bleu_score["bleu"],
        "BERTScore_Precision": bertscore_results["precision"],
        "BERTScore_Recall": bertscore_results["recall"],
        "BERTScore_F1": bertscore_results["f1"]
    }

    # Export to CSV
    print(f"▶ Exporting evaluation report to {REPORT_OUTPUT}...")
    results_df = pd.DataFrame([results])
    results_df.to_csv(REPORT_OUTPUT, index=False)

    # Print summary
    print("\n▶ Evaluation Summary:")
    for metric, value in results.items():
        print(f"  {metric}: {value:.4f}")

    return results

if __name__ == "__main__":
    # Replace with your actual test dataset path
    evaluate_model(test_data_path="datasets/test_data.json")