import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from flask import Flask, request, render_template

# Initialize Flask app
app = Flask(__name__)

# Configuration
MODEL_PATH = "trained_models/llama-2-13b-chat-lora-final_merged"
CACHE_DIR = "huggingface_cache"
MAX_LENGTH = 256

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

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        cache_dir=CACHE_DIR,
        local_files_only=True,
        use_fast=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        device_map="auto",
        cache_dir=CACHE_DIR,
        local_files_only=True,
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16
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