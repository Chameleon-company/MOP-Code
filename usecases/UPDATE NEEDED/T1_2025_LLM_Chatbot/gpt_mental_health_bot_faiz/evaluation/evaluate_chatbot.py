import json
import random
import os
from tqdm import tqdm
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

import evaluate
import nltk
nltk.download("punkt")

# Load OpenAI API key
load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")
if not openai_key:
    raise ValueError(" OpenAI API key missing in .env")

# Load cleaned dataset
with open("cleaned_data/mentalchat16k_cleaned.json", "r") as f:
    data = json.load(f)

#  Sample a subset for evaluation
sample_data = random.sample(data, 100)

# 💬 Setup the chatbot
chat = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7)

# Use the original dataset's prompt as system message
system_prompt = (
    "You are a helpful mental health counselling assistant, "
    "please answer the mental health questions based on the patient's description.\n"
    "The assistant gives helpful, comprehensive, and appropriate answers to the user's questions."
)

# 🧾 Storage for evaluation
predictions = []
references = []

print(" Generating responses from GPT-3.5 for 100 samples...\n")

for item in tqdm(sample_data):
    question = item["question"]
    reference = item["answer"]

    try:
        # Invoke GPT-3.5 using LangChain
        response = chat.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=question)
        ])
        predictions.append(response.content)
        references.append(reference)

    except Exception as e:
        print("Error during generation:", e)
        continue

# Evaluate using standard NLP metrics
print("\n Evaluating chatbot response quality...\n")

# BLEU
bleu = evaluate.load("bleu")
bleu_result = bleu.compute(predictions=predictions, references=[[ref] for ref in references])

# ROUGE
rouge = evaluate.load("rouge")
rouge_result = rouge.compute(predictions=predictions, references=references)

# BERTScore
bertscore = evaluate.load("bertscore")
bertscore_result = bertscore.compute(predictions=predictions, references=references, lang="en")


#use another rouge


#  Print results
print(f" BLEU Score: {bleu_result['bleu']:.4f}")
print(f" ROUGE-L Score: {rouge_result['rougeL']:.4f}")
print(f" BERTScore (F1): {sum(bertscore_result['f1']) / len(bertscore_result['f1']):.4f}")
