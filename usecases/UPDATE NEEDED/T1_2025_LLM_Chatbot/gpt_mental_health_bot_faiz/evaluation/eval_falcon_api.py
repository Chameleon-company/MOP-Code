import os
import json
import evaluate
import nltk
import requests
import csv
from dotenv import load_dotenv
from tqdm import tqdm
from datetime import datetime

nltk.download("punkt")

# Load API token
load_dotenv()
hf_token = os.getenv("HF_API_TOKEN")
if not hf_token:
    raise Exception("Hugging Face API token not found in .env file.")

headers = {
    "Authorization": f"Bearer {hf_token}"
}
API_URL = "https://api-inference.huggingface.co/models/tiiuae/falcon-7b-instruct"

# System prompt
system_prompt = (
    "You are a compassionate and helpful mental health assistant.\n"
    "Your role is to provide emotional support and practical guidance to users struggling with anxiety, depression, or related issues.\n"
    "Your tone is empathetic, calm, non-judgmental, and professional.\n"
    "Answer each concern thoughtfully and thoroughly.\n"
)

# Load dataset
with open("evaluation_datasets/colab_gpu_limited_100.json", "r") as f:
    data = json.load(f)

# Load evaluation metrics
bleu = evaluate.load("bleu")
rouge = evaluate.load("rouge")
bertscore = evaluate.load("bertscore")

# CSV logging setup
os.makedirs("logs", exist_ok=True)
csv_path = "logs/falcon7b_results.csv"
write_header = not os.path.exists(csv_path)

with open(csv_path, mode="a", newline="") as csvfile:
    writer = csv.writer(csvfile)
    
    if write_header:
        writer.writerow(["timestamp", "temperature", "BLEU", "ROUGE-1", "ROUGE-2", "ROUGE-L", "BERTScore_F1"])

    for temp in [0.3, 0.5, 0.7, 0.9]:
        predictions = []
        references = []

        print(f"\n🚀 Generating responses with temperature = {temp}...\n")

        for pair in tqdm(data):
            prompt = f"{system_prompt}\nUser: {pair['question']}\nAssistant:"
            payload = {
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens": 300,
                    "temperature": temp
                }
            }

            try:
                response = requests.post(API_URL, headers=headers, json=payload)
                result = response.json()

                if isinstance(result, dict) and "error" in result:
                    print(f"API Error: {result['error']}")
                    predictions.append("")
                else:
                    reply = result[0]["generated_text"].split("Assistant:")[-1].strip()
                    predictions.append(reply)

                references.append(pair["answer"])

            except Exception as e:
                print(f"❌ Request failed: {e}")
                predictions.append("")
                references.append("")

        print(f"\n📊 Evaluating temperature = {temp}...\n")
        bleu_result = bleu.compute(predictions=predictions, references=[[ref] for ref in references])
        rouge_result = rouge.compute(predictions=predictions, references=references)
        bert_result = bertscore.compute(predictions=predictions, references=references, lang="en")

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        writer.writerow([
            timestamp,
            temp,
            round(bleu_result['bleu'], 4),
            round(rouge_result['rouge1'], 4),
            round(rouge_result['rouge2'], 4),
            round(rouge_result['rougeL'], 4),
            round(sum(bert_result['f1']) / len(bert_result['f1']), 4)
        ])

        print(f"✅ Results logged for temperature = {temp}\n")
