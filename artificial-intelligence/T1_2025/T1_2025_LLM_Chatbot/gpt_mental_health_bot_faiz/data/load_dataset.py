from datasets import load_dataset
import json
import os

# Loading the  MentalChat16K dataset
print("Downloading MentalChat16K...")
ds = load_dataset("ShenLab/MentalChat16K")
dataset = ds["train"]

# checking the output directory
os.makedirs("data", exist_ok=True)

# Save the dataset as Json
with open("data/mentalchat16k_raw.json", "w") as f:
    for example in dataset:
        json.dump(example, f)
        f.write("\n")

print(f" Dataset saved with {len(dataset)} entries.")
