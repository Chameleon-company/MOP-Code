import json

# Load your data
with open('no_gpu_limit_500.json', 'r') as f:
    data = json.load(f)

# Convert data to OpenAI's expected format
formatted_data = []

for item in data:
    formatted_item = {
        "messages": [
            {"role": "system", "content": "You are a compassionate and helpful mental health assistant. You provide emotional support and guidance for users experiencing anxiety and depression."},
            {"role": "user", "content": item["question"]},
            {"role": "assistant", "content": item["answer"]}
        ]
    }
    formatted_data.append(formatted_item)

# Save the formatted data
with open('mentalchat16k_openai.jsonl', 'w') as f:
    for entry in formatted_data:
        f.write(json.dumps(entry) + '\n')

print(f"✅ Done! {len(formatted_data)} entries written to mentalchat16k_openai.jsonl")
