import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from flask import Flask, request, render_template

# Initialize Flask app
app = Flask(__name__)

# Configuration
MODEL_PATH = "D:/version_2.0_250402/newoutput/llama-2-7b-chat-lora-final_merged"  # Use 7B model for CPU
CACHE_DIR = "D:/huggingface_cache"
MAX_LENGTH = 256

# Device setup (force CPU)
def set_device():
    device = torch.device("cpu")
    print("▶ Using CPU.")
    return device

# Load the model and tokenizer
def load_model_and_tokenizer():
    device = set_device()
    print(f"▶ Loading model and tokenizer from {MODEL_PATH}...")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        cache_dir=CACHE_DIR,
        local_files_only=True,
        use_fast=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        device_map="cpu",  # Force CPU
        cache_dir=CACHE_DIR,
        local_files_only=True,
        torch_dtype=torch.float16,  # Use FP16 to reduce memory usage
        low_cpu_mem_usage=True
    ).to(device)
    model.eval()

    return model, tokenizer, device

# Generate a response from the model
def generate_response(model, tokenizer, device, user_input):
    input_text = f"[INST] {user_input} [/INST]"
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=MAX_LENGTH).to(device)

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

# Load the model once when the app starts
model, tokenizer, device = load_model_and_tokenizer()

# Flask routes
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        user_input = request.form["user_input"]
        if user_input.lower() == "exit":
            return render_template("index.html", conversation="Chatbot: Goodbye!")
        
        response = generate_response(model, tokenizer, device, user_input)
        conversation = f"You: {user_input}\nChatbot: {response}"
        return render_template("index.html", conversation=conversation)
    
    return render_template("index.html", conversation="")

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)