from openai import OpenAI
from llm_module.config import API_KEY
from llm_module.rag_engine import load_rules

client = OpenAI(api_key=API_KEY)

def generate_reasoning(data):
    rules = load_rules()

    prompt = f"""
    You are a civil engineer.

    Rules:
    {rules}

    Crack:
    Length: {data['crack_length']}
    Width: {data['crack_width']}
    Location: {data['location']}

    Explain:
    - Why this crack is dangerous or not
    - Risks involved
    """

    res = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"user","content":prompt}]
    )

    return res.choices[0].message.content