import pandas as pd
from llama_cpp import Llama
import os

# loading the cleaned dataset
df = pd.read_csv("datasets/cleaned_empathetic_dataset.csv")

# loading Mistral Model
model_path = "models/mistral-7b-instruct-v0.1-q4_k_m.gguf"
llm = Llama(
    model_path=model_path,
    n_gpu_layers=16,
    n_ctx=2048,
    n_threads=8,
    use_mlock=True,
    verbose=False
)

# starting session
print("\n Welcome to the Mental Health Chatbot")
print("Type your message or 'sample' to test a dataset prompt.")
print("Type 'exit' to quit.\n")

while True:
    user_input = input("You: ").strip()

    if user_input.lower() in ['exit', 'quit']:
        print("Goodbye!\n")
        break

    if user_input.lower() == 'sample':
        sample = df.sample(1).iloc[0]
        prompt = sample["Situation"]
        emotion = sample["emotion"]
        expected = sample["labels"]

        print(f"\nDataset Prompt: {prompt}")
        print(f"Emotion: {emotion}")
        print(f"Expected: {expected}")

        formatted_prompt = f"<s>[INST] You are a kind and supportive mental health assistant.\nEmotion: {emotion}\n{prompt} [/INST]"
    else:
        formatted_prompt = f"<s>[INST] You are a kind and supportive mental health assistant.\n{user_input} [/INST]"

    response = llm(formatted_prompt, max_tokens=200, stop=["</s>", "User:", "\n\n"])
    reply = response["choices"][0]["text"].strip()

    print(f"\nMistral: {reply}\n")
