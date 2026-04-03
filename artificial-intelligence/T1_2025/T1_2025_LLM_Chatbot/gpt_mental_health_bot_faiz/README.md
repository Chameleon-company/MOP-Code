# 🧠 Mental Health Chatbot

Welcome! This project is a mental health support chatbot using both **GPT-3.5 Turbo** (via LangChain and OpenAI) and **Falcon 7B** (via Hugging Face Inference API). The chatbot provides empathetic responses to mental health queries and supports model evaluation and hyperparameter tuning.

---

## 📁 Project Structure

mental_health_chatbot/
│
├── chatbot/
│   ├── langchain_app.py       # GPT-3.5 chatbot using LangChain
│   └── falcon_chatbot.py      # Falcon 7B chatbot using Hugging Face API
│
├── cleaned_data/              # Full cleaned dataset
├── data/                      # Scripts to clean and reduce datasets
├── evaluation/                # Evaluation scripts for GPT and Falcon
├── evaluation_datasets/      # 100 & 500 sample datasets
├── logs/                      # Evaluation logs and results
│
├── model/                     # (Reserved for future fine-tuning)
├── .env                       # API keys (user-created)
├── requirements.txt           # All required dependencies
├── README.md                  # This file
└── venv/                      # Your virtual environment (not needed for sending)



## 🛠️ Setup Instructions

### 1. 🗂️ Unzip the Project
Unzip the file `mental_health_chatbot.zip` and navigate into the folder:
```bash
cd mental_health_chatbot
```

### 2. 🐍 Create a Virtual Environment
bash
python3 -m venv venv
source venv/bin/activate      # macOS/Linux
venv\Scripts\activate         # Windows


### 3. 📦 Install Required Packages
```bash
pip install -r requirements.txt
```

### 4. 🔐 Add API Keys
Create a `.env` file in the project root:
```
OPENAI_API_KEY=your_openai_api_key
HF_API_TOKEN=your_huggingface_api_token
```

---

## 💬 Running the Chatbot

### ▶️ GPT-3.5 Chatbot (LangChain)
```bash
python chatbot/langchain_app.py
```
This uses your OpenAI GPT-3.5 Turbo API and runs completely via cloud (no GPU required).

---
### Flask Web Chatbot

### use python app.py --port 5016 



### ▶️ Falcon 7B Chatbot (Hugging Face API)
```bash
python chatbot/falcon_chatbot.py
```
This uses Falcon-7B remotely via Hugging Face's **Inference API**. No model is downloaded locally — it's cloud-only. Ensure your Hugging Face account has inference credits.

---

## 📊 Evaluation

### ✅ Evaluate GPT-3.5 with Different Hyperparameters
```bash
python evaluation/eval_hyperparams_gpt35.py
```

### ✅ Evaluate Falcon 7B Performance
```bash
python evaluation/eval_falcon_api.py
```

Results are logged automatically in the `logs/` folder in CSV format.

---


### to run the flask app
### python app.py --port=5002


## 🧪 Datasets

- `cleaned_data/mentalchat16k_cleaned.json`: Full cleaned dataset
- `evaluation_datasets/colab_gpu_limited_100.json`: Use for limited GPU or API budget
- `evaluation_datasets/no_gpu_limit_500.json`: More comprehensive, slower

---

## 📎 Notes

- You **don’t need a GPU**. All models run via API.
- The Falcon model is **not downloaded**. It uses the Hugging Face cloud directly.
- You can view logs and results in the `logs/` folder, including CSV scores for BLEU, ROUGE, and BERTScore.

---

## 🙋 Need Help?

If you have any trouble running the project, feel free to reach out via email or message Faiz.

---

```
