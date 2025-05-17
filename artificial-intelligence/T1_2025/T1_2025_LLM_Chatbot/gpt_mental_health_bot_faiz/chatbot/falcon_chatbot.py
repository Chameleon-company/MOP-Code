import os
import requests
from dotenv import load_dotenv
from datetime import datetime

# Loading the token
load_dotenv()
hf_token = os.getenv("HF_API_TOKEN")
print("Token loaded:", os.getenv("HF_API_TOKEN"))

if not hf_token:
    raise Exception("Hugging Face API token not found in .env file.")

# API Endpoint for Falcon 7B hosted on Hugging Face
API_URL = "https://api-inference.huggingface.co/models/tiiuae/falcon-7b-instruct"
headers = {
    "Authorization": f"Bearer {hf_token}"
}

# system prompt
system_prompt = (
    "You are a compassionate and helpful mental health assistant.\n"
    "Your role is to provide emotional support and practical guidance to users struggling with anxiety, depression, or related issues.\n"
    "Your tone is empathetic, calm, non-judgmental, and professional.\n"
    "Answer each concern thoughtfully and thoroughly.\n"
)

log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "chat_log.txt")

# UI Header
print("=" * 60)
print("|{:^58}|".format("Mental Health Chatbot (Falcon 7B - Cloud API Only)"))
print("=" * 60)
print("Type 'exit' or 'quit' to end the conversation.\n")

# Chat loop
while True:
    user_input = input("You: ").strip()
    if user_input.lower() in ["exit", "quit", "bye"]:
        print("\n" + "-" * 60)
        print("👋 Take care. You’re not alone.")
        print("-" * 60 + "\n")
        break

    prompt = f"{system_prompt}\nUser: {user_input}\nAssistant:"

    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 300,
            "temperature": 0.7
        }
    }

    try:
        response = requests.post(API_URL, headers=headers, json=payload)
        result = response.json()

        if isinstance(result, dict) and "error" in result:
            print(f"API Error: {result['error']}")
            continue

        reply = result[0]["generated_text"].split("Assistant:")[-1].strip()

        print("\n" + "-" * 60)
        print("Bot:")
        print(reply)
        print("-" * 60 + "\n")

        # Log
        with open(log_file, "a") as f:
            f.write(f"[{datetime.now()}] USER: {user_input}\n")
            f.write(f"[{datetime.now()}] BOT: {reply}\n\n")

    except Exception as e:
        print(f"Request failed: {e}")
