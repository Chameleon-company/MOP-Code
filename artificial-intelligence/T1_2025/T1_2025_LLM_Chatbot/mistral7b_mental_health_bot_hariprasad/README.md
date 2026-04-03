
# Mental Health Chatbot (Mistral-7B)

This project is an AI-powered chatbot designed to provide empathetic, supportive responses to individuals dealing with anxiety, stress and emotional challenges. It is fine-tuned from **mistralai/Mistral-7B-Instruct-v0.2** using real-world mental health datasets. The chatbot runs via Google Colab with options for Streamlit UI and a connected Android app using a Flask API.

---

## Features

- Fine-tuned **Mistral-7B-Instruct-v0.2** using real & synthetic data
- Supports **long/short response mode**
- Built-in **safety filters** for crisis keywords
- Deployment options:
  - 📱 Android Kotlin app with Flask backend
  - 🟣 Streamlit web UI on Colab
- Optimized inference with **4-bit quantization** using `bitsandbytes`

---


## Base Model

| Model | Link |
|-------|------|
| `mistralai/Mistral-7B-Instruct-v0.2` | [Hugging Face](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2) |

---

## Datasets Used

| Dataset | Description | Link |
|---------|-------------|------|
| **CounselChat** | Q&A from licensed therapists on common emotional issues | [HF - CounselChat](https://huggingface.co/datasets/nbertagnolli/counsel-chat) |
| **EmpatheticDialogues** | Facebook AI dataset of 25k conversations designed to train empathetic models | [HF - EmpatheticDialogues](https://huggingface.co/datasets/facebook/empathetic_dialogues) |
| **Synthetic Data** | GPT-generated mental health instruction-output pairs to improve empathy diversity | *Internal* |

✅ All datasets were cleaned, deduplicated and merged using custom preprocessing.

---

## Android Integration

- Kotlin app with `Retrofit` to call `/chat` endpoint
- Includes:
  - Voice input
  - Typing animation
  - Short/Long response toggle
  - Avatar-based UI
- Backend runs via Flask + ngrok in Colab

---

## Safety Filter Logic

Detects phrases like:
- *"I want to die"*
- *"kill myself"*
- *"not worth living"*

Auto-responds with:

> 💙 I'm really sorry you're feeling this way. You're not alone.  
> Please seek help from a professional. Contact Lifeline Australia at 13 11 14 (24/7).

---

## Inference Optimized For

| Hardware | Spec |
|----------|------|
| Colab Pro | T4/A100 GPU (4-bit quantized model) |
| Format | Transformers + PEFT (LoRA) |
| Max tokens | 200–400 |

---
