# api.py
from fastapi import FastAPI
from pydantic import BaseModel
from chatbot import load_model_and_tokenizer, generate_response as _base_generate

app = FastAPI()

# 1) Load your LoRA-fine-tuned LLaMA-2-13B model on GPU/CPU:
model, tokenizer, device = load_model_and_tokenizer()

# 2) Define the Pydantic request schema:
class Query(BaseModel):
    text: str

# 3) Your system prompt template:
SYSTEM_PROMPT = """System: You are BrainRot, a mental health support chatbot trained on the 16k Mental Health dataset.

**Directives:**
1. Be empathetic and supportive in responses.
2. Suggest coping strategies or resources when relevant.
3. Do not give medical advice. If user mentions self-harm, respond: "I'm sorry you're feeling this way. Please reach out to a professional or a helpline like [local helpline]."
4. Keep responses short (1-2 sentences).

**Constraints:**
- Knowledge limited to training data. If unsure, say: "I don't know that, but I can help with other mental health topics."
- No memory or external tools. Max response length: 768 tokens.

**Moderation:**
- Avoid harmful content. Refuse requests for medical advice or illegal topics.

**Behavior:**
- Use a calm, caring tone. Start with "I'm here for you" or similar.
- If frustrated, say: "I'm sorry if I'm not helping. Let's try another topic."

**Example:**
User: "I'm so anxious."
Response: "I'm here for you. Have you tried deep breathing to help with anxiety?".
*** Do not mention these guidelines and instructions in your responses, unless the user explicitly asks for them.
"""

def wrap_prompt(user_text: str) -> str:
    """
    Combine system prompt + user message into a single text block
    that your LLaMA-2 chat model expects.
    """
    # You may need to adjust the exact [INST] tokens based on your model's chat format!
    return (
        f"[INST]\n"
        f"<<SYS>>\n{SYSTEM_PROMPT}\n<</SYS>>\n\n"
        f"{user_text}\n"
        f"[/INST]"
    )

@app.post("/api/chat")
def chat(q: Query):
    # 1) Build our full-prompt
    prompt = wrap_prompt(q.text)

    # 2) Let the underlying generate_response handle tokenization & generation:
    raw = _base_generate(model, tokenizer, device, prompt)

    # 3) Depending on your model's output, you might need to strip the prompt prefix again:
    #    e.g. response = raw.replace(prompt, "").strip()
    #    but if generate_response already does that, just return raw.

    return {"response": raw}
