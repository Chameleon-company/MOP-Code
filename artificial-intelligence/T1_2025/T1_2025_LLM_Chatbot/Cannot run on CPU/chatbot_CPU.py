import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Configuration
MODEL_PATH = "D:/version_2.0_250402/newoutput/llama-2-7b-chat-lora-final_merged"  # Use 7B model for CPU
CACHE_DIR = "D:/huggingface_cache"
MAX_LENGTH = 256  # Adjust based on memory constraints

# Device setup (force CPU)
def set_device():
    device = torch.device("cpu")
    print("▶ Using CPU.")
    return device

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
            device_map="cpu",  # Force CPU
            cache_dir=CACHE_DIR,
            local_files_only=True,
            torch_dtype=torch.float16,  # Use FP16 to reduce memory usage
            low_cpu_mem_usage=True  # Optimize for CPU memory
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