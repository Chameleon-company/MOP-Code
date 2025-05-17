import pandas as pd
from llama_cpp import Llama

# Load dataset
df = pd.read_csv("datasets/cleaned_empathetic_dataset.csv")
df = df.drop_duplicates(subset=["Situation"], keep="first")
samples = df.sample(5)

# Load model
llm = Llama(model_path="models/mistral-7b-instruct-v0.1-q4_k_m.gguf",
            n_gpu_layers=16, n_ctx=2048, n_threads=8, use_mlock=True)

results = []

for _, row in samples.iterrows():
    prompt = row["Situation"]
    expected = row["labels"]
    emotion = row["emotion"]
    reply = llm(prompt, max_tokens=150, stop=["User:", "\n\n"])["choices"][0]["text"].strip()

    results.append({
        "Prompt": prompt,
        "Emotion": emotion,
        "Expected": expected,
        "Mistral Response": reply
    })

df_out = pd.DataFrame(results)
df_out.to_csv("logs/mistral_log.csv", index=False)
print("Responses saved to logs/mistral_log.csv")
