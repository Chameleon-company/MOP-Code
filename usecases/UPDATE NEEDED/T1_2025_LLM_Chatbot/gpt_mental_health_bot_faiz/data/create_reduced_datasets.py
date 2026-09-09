import json
import random
import os

# Load full cleaned dataset
with open("cleaned_data/mentalchat16k_cleaned.json", "r") as f:
    full_data = json.load(f)

# Shuffle for random sampling
random.shuffle(full_data)

# Create smaller versions
no_gpu_limit_data = full_data[:500]
gpu_limited_data = full_data[:100]

# Create output folder
os.makedirs("evaluation_datasets", exist_ok=True)

# Save files
with open("evaluation_datasets/no_gpu_limit_500.json", "w") as f:
    json.dump(no_gpu_limit_data, f, indent=2)

with open("evaluation_datasets/colab_gpu_limited_100.json", "w") as f:
    json.dump(gpu_limited_data, f, indent=2)

print("✅ Created 500 and 100 row datasets for evaluation.")
