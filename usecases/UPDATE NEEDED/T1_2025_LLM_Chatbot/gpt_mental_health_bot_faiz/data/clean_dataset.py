import json
import re
from tqdm import tqdm
import os

# Load the raw JSONL file
input_file = "data/mentalchat16k_raw.jsonl"
output_file = "cleaned_data/mentalchat16k_cleaned.json"

# i am checking that the directory is present 
os.makedirs("cleaned_data", exist_ok=True)

# Basic PII removal function
#i used regex to filter out confidential information
def remove_pii(text):
    # Remove the emails
    text = re.sub(r'\S+@\S+', '[EMAIL]', text)
    # Removing the urls
    text = re.sub(r'http\S+|www\.\S+', '[URL]', text)
    # RemovING THE  phone numbers
    text = re.sub(r'\b\d{10,}\b', '[PHONE]', text)
    return text.strip()

# cleaned data is being stored
seen_pairs = set()
cleaned_data = []

with open(input_file, 'r') as f:
    for line in tqdm(f, desc="Cleaning"):
        line = line.strip()
        if not line:
            continue  #empty lines will be skipped
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue  # Skipping any malformed lines

        if not isinstance(entry, dict):
            continue  # Skipping if not dict

        inp = entry.get("input", "")
        out = entry.get("output", "")

        if not isinstance(inp, str) or not isinstance(out, str):
            continue  # Skipping if input/output are not strings

        inp = inp.strip()
        out = out.strip()

        if not inp or not out:
            continue  # Skipping  empty input/output

        inp = remove_pii(inp)
        out = remove_pii(out)

        if len(out.split()) > 300:
            continue

        pair = (inp, out)
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)

        cleaned_data.append({
            "question": inp,
            "answer": out
        })


# Save the cleaned data to JSON
with open(output_file, "w") as f:
    json.dump(cleaned_data, f, indent=2)

print(f"\n✅ Cleaned and saved {len(cleaned_data)} unique Q&A pairs to '{output_file}'")
