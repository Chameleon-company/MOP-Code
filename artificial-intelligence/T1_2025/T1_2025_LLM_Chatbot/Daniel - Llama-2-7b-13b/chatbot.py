import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Configuration
MODEL_PATH = "trained_models/llama-2-13b-chat-lora-final_merged"  # Path to your fine-tuned 13B model
CACHE_DIR = "huggingface_cache"
MAX_LENGTH = 768  # Adjust based on memory constraints (256, 512, 765, 1024, 1280, 1536, ...)

# Device setup
def set_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("▶ Using GPU:", torch.cuda.get_device_name(0))
        torch.cuda.empty_cache()
        return device
    else:
        print("▶ No GPU available, using CPU.")
        return torch.device("cpu")

# Load the model and tokenizer
def load_model_and_tokenizer():
    device = set_device()
    print(f"▶ Loading model and tokenizer from {MODEL_PATH}...")

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_PATH,
            cache_dir=CACHE_DIR,
            local_files_only=True,
            use_fast=True
        )
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        raise

    try:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            device_map="auto",
            cache_dir=CACHE_DIR,
            local_files_only=True,
            load_in_4bit=True,  # Use 4-bit quantization
            bnb_4bit_compute_dtype=torch.float16
        ).to(device)
        model.eval()
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

    return model, tokenizer, device

# Generate a response from the model
def generate_response(model, tokenizer, device, user_input):
    # Format the input as per the model's expected prompt
    input_text = f"[INST] {user_input} [/INST]"
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=MAX_LENGTH).to(device)

    try:
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=MAX_LENGTH,
                num_return_sequences=1,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True).replace(input_text, "").strip()
        return response
    except Exception as e:
        print(f"Error generating response: {e}")
        return "Sorry, I encountered an error while generating a response."

# Main chatbot loop
def run_chatbot():
    model, tokenizer, device = load_model_and_tokenizer()
    print("▶ Chatbot is ready! Type 'exit' to quit.")

    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("▶ Goodbye!")
            break

        response = generate_response(model, tokenizer, device, user_input)
        print(f"Bot: {response}")

if __name__ == "__main__":
    run_chatbot()