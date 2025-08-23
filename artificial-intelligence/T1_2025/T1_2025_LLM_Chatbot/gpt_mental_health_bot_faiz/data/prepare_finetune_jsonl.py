import json

# Load cleaned Q&A pairs
with open("cleaned_data/mentalchat16k_cleaned.json", "r") as f:
    raw_data = json.load(f)

# Format data for fine-tuning
formatted = []
for pair in raw_data:
    formatted.append({
        "messages": [
            {"role": "system", "content": "You are a helpful mental health counselling assistant. The assistant gives helpful, comprehensive, and appropriate answers to the user's questions."},
            {"role": "user", "content": pair["question"]},
            {"role": "assistant", "content": pair["answer"]}
        ]
    })

# Save to JSONL
with open("cleaned_data/mentalchat16k_finetune.jsonl", "w") as f:
    for item in formatted:
        f.write(json.dumps(item) + "\n")

print(f" Saved {len(formatted)} entries to mentalchat16k_finetune.jsonl")
