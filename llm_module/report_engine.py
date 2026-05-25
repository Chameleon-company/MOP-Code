from openai import OpenAI
from llm_module.config import API_KEY
from llm_module.severity import classify_severity
from llm_module.rag_engine import load_rules

# ✅ FIXED CLIENT
client = OpenAI(api_key=API_KEY)

def generate_report(data):
    severity = classify_severity(data['crack_length'], data['crack_width'])
    rules = load_rules()

    prompt = f"""
    Rules:
    {rules}

    Crack:
    Length: {data['crack_length']}
    Width: {data['crack_width']}
    Location: {data['location']}
    Severity: {severity}

    Provide:
    - Risk
    - Recommendation
    """

    res = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    return f"Severity: {severity}\n" + res.choices[0].message.content