import os
import json
import csv
from datetime import datetime
from tqdm import tqdm
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
import evaluate
import nltk

# 📦 Download tokenizer for BLEU
nltk.download("punkt")

# 🔐 Load OpenAI API key
load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")
if not openai_key:
    raise Exception("❌ OpenAI API key not found in .env!")

# 📂 Load 100-row dataset
with open("evaluation_datasets/colab_gpu_limited_100.json", "r") as f:
    data = json.load(f)

# 🧠 System prompt
system_prompt = (
    "You are a compassionate and helpful mental health assistant.\n"
    "You provide emotional support and guidance for users experiencing anxiety and depression.\n"
    "Be empathetic, calm, and non-judgmental in your responses."
)

# 🔁 Hyperparameter configs to evaluate
hyperparam_sets = [
    {"temperature": 0.3, "top_p": 1.0, "frequency_penalty": 0.0, "presence_penalty": 0.0},
    {"temperature": 0.5, "top_p": 0.95, "frequency_penalty": 0.5, "presence_penalty": 0.0},
    {"temperature": 0.7, "top_p": 0.9, "frequency_penalty": 0.0, "presence_penalty": 0.5},
    {"temperature": 0.9, "top_p": 0.85, "frequency_penalty": 0.5, "presence_penalty": 0.5},
]

# 📊 Load evaluation metrics
bleu = evaluate.load("bleu")
rouge = evaluate.load("rouge")
bertscore = evaluate.load("bertscore")

# 📁 Set up log file
log_file = "logs/hparam_results.csv"
os.makedirs("logs", exist_ok=True)

if not os.path.exists(log_file):
    with open(log_file, "w") as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp", "temperature", "top_p", "freq_penalty", "pres_penalty",
            "BLEU", "ROUGE-1", "ROUGE-2", "ROUGE-L", "BERTScore_F1"
        ])

# 🧾 Store all results for terminal display
all_results = []

# 🔁 Loop through each config
for config in hyperparam_sets:
    print(f"\n🚀 Testing config: {config}")

    chat = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        temperature=config["temperature"],
        top_p=config["top_p"],
        frequency_penalty=config["frequency_penalty"],
        presence_penalty=config["presence_penalty"]
    )

    predictions = []
    references = []

    # 🧠 Generate predictions
    for pair in tqdm(data, desc="Generating Responses"):
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=pair["question"])
        ]
        try:
            response = chat.invoke(messages)
            predictions.append(response.content)
            references.append(pair["answer"])
        except Exception as e:
            print(f"❌ Error: {e}")
            predictions.append("")
            references.append("")

    # 📊 Evaluate
    print("📊 Computing BLEU...")
    bleu_result = bleu.compute(predictions=predictions, references=[[r] for r in references])

    print("📊 Computing ROUGE...")
    rouge_result = rouge.compute(predictions=predictions, references=references)

    print("📊 Computing BERTScore (roberta-large)...")
    bert_result = bertscore.compute(
        predictions=predictions,
        references=references,
        lang="en",
        model_type="roberta-large"
    )

    avg_bert_score = round(sum(bert_result["f1"]) / len(bert_result["f1"]), 4)

    # Save results
    result_row = {
        "temperature": config["temperature"],
        "top_p": config["top_p"],
        "freq_penalty": config["frequency_penalty"],
        "pres_penalty": config["presence_penalty"],
        "BLEU": round(bleu_result["bleu"], 4),
        "ROUGE-1": round(rouge_result["rouge1"], 4),
        "ROUGE-2": round(rouge_result["rouge2"], 4),
        "ROUGE-L": round(rouge_result["rougeL"], 4),
        "BERTScore_F1": avg_bert_score
    }

    all_results.append(result_row)

    # Write to CSV
    with open(log_file, "a") as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            *result_row.values()
        ])

    print("✅ Logged results for config:", config)

# 🖨️ Print summary table
print("\n📊 Summary of all hyperparameter results:\n")
print("{:<5} {:<6} {:<6} {:<6} {:<6} {:<8} {:<8} {:<8} {:<8} {:<10}".format(
    "Temp", "TopP", "Freq", "Pres", "BLEU", "ROUGE-1", "ROUGE-2", "ROUGE-L", "BERTScore", ""
))
print("-" * 90)
for res in all_results:
    print("{:<5} {:<6} {:<6} {:<6} {:<6} {:<8} {:<8} {:<8} {:<8} {:<10}".format(
        res["temperature"], res["top_p"], res["freq_penalty"], res["pres_penalty"],
        res["BLEU"], res["ROUGE-1"], res["ROUGE-2"], res["ROUGE-L"], res["BERTScore_F1"], ""
    ))
