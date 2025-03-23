
# Mental Health Chatbot

This project is an AI-powered chatbot designed to provide supportive, empathetic responses for individuals experiencing anxiety and depression. It uses the **Mistral 7B Instruct** model in **GGUF format** and runs locally via `llama-cpp-python`, optimized for systems with limited GPU resources (like RTX 3050 Ti with 4GB VRAM).

---

## Features in v1

- **Terminal-based chatbot** powered by Mistral 7B Instruct
- **Empathetic response generation** using a cleaned mental health dataset
- **Sample-based mode** to test real-world prompts from the dataset
- **Model evaluation script** for generating and logging model responses
- Dataset cleaning and deduplication
- Lightweight, local deployment — works on mid-range laptops

---

## Project Structure

```
CHATBOT/
├── datasets/
│   ├── raw/                                # Raw dataset (e.g., emotion-emotion_69k.csv)
│   └── cleaned_empathetic_dataset.csv      # Cleaned dataset
├── models/                                 # GGUF model files (e.g., Mistral)
├── logs/                                   # Output logs for evaluation results
├── notebooks/
│   └── data-cleanr.ipynb                    # Jupyter notebook for data preprocessing
├── chatbot_terminal.py                     # Terminal chatbot script
├── evaluate_model.py                       # Script to test model on sample prompts
├── README.md                               # Project documentation (this file)
```

---

## Setup Instructions

### 1. Clone the Repository & Set Up Environment

```bash
conda create -n chatbot_env python=3.9
conda activate chatbot_env
pip install pandas llama-cpp-python
```
---

### 2. Download Mistral 7B Model (GGUF)

Place a quantized `.gguf` file into the `models/` folder:

```
models/mistral-7b-instruct-v0.1-q4_k_m.gguf
```

Model available in [Hugging Face](https://huggingface.co).

---

### 3. Run the Terminal Chatbot

```bash
python chatbot_terminal.py
```

Options:
- Type your message for a custom response
- Type `sample` to test a prompt from the dataset
- Type `exit` to quit

---

### 4. Evaluate Model Responses

```bash
python evaluate_model.py
```

This will:
- Load 5 random samples from the dataset
- Send each to the model
- Save responses to `logs/mistral_log.csv`

You can change the sample size in the script to test more examples.

---

## Prompt Format

The chatbot uses **instruction-style prompting** expected by Mistral 7B:

```
<s>[INST] You are a kind and supportive mental health assistant.
I'm feeling very anxious lately. [/INST]
```

This ensures the model responds with empathy, calmness, and support.

---

## Dataset Details

We used a cleaned version of the `emotion-emotion_69k.csv` dataset, focusing on:
- Real-world mental health scenarios
- Associated emotion labels (e.g., anxiety, depression, grief)
- Expected supportive responses

Preprocessing was done in `notebooks/data-cleanr.ipynb`, and the final cleaned file is stored in:
```
datasets/cleaned_empathetic_dataset.csv
```

---

## Optimized for Local Inference

This project is configured to work efficiently on systems with:

| Component     | Minimum Required   |
|---------------|--------------------|
| GPU           | NVIDIA GPU with 4GB VRAM (e.g., RTX 3050 Ti) |
| RAM           | 16 GB              |
| CPU Threads   | 6–8 recommended    |
| Model Format  | GGUF (Quantized, e.g., q4_k_m) |

Settings like `n_gpu_layers`, `n_ctx`, and `n_threads` can be tuned in the script to match your system.

---

## Evaluation Example Output

Sample output from `evaluate_model.py`:

| Prompt | Emotion | Expected | Mistral Response |
|--------|---------|----------|------------------|
| "I've been feeling overwhelmed..." | anxiety | supportive advice | empathetic response |

Responses are saved to:
```
logs/mistral_log.csv
```

---

## Disclaimer

This chatbot is intended for educational and research purposes only. It is **not a substitute for professional mental health advice or treatment.**

---

## License

This project is open-source and licensed for non-commercial, educational use.

---
