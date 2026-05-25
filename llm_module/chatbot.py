from openai import OpenAI
from llm_module.config import API_KEY
from llm_module.rag_engine import load_rules

# Initialize client
client = OpenAI(api_key=API_KEY)


def chatbot_response(question, data):
    """
    Advanced chatbot:
    - Uses RAG (rules)
    - Dynamic response depth
    - Structured engineering answers
    """

    # 🔹 Load domain rules (RAG)
    rules = load_rules()

    # 🔥 SYSTEM PROMPT (INTELLIGENT + DYNAMIC)
    system_prompt = f"""
You are a professional civil engineering assistant for infrastructure crack analysis.

DOMAIN RULES:
{rules}

INTELLIGENT RESPONSE BEHAVIOR:

1. Understand the question type:
   - Basic → (location, type, severity)
   - Analytical → (why, cause, impact)
   - Decision → (risk, what to do)

2. Adjust explanation depth dynamically:
   - Basic → short + explanation
   - Analytical → deeper reasoning
   - Decision → reasoning + recommendation

3. ALWAYS structure answers:
   - Direct Answer
   - Explanation
   - (Optional) Impact / Recommendation

4. Use crack data properly:
   - Length → spread of damage
   - Width → severity indicator
   - Location → crack type context

5. Avoid:
   - Generic answers
   - Repeating full reports
   - Irrelevant content

6. Keep tone:
   - Professional
   - Engineering-focused
   - Realistic (like industry system)
"""

    # 🔹 USER PROMPT
    user_prompt = f"""
User Question:
{question}

Crack Details:
- Length: {data['crack_length']} m
- Width: {data['crack_width']} mm
- Location: {data['location']}
- Severity: {data['severity']}

Provide a structured and context-aware answer.
"""

    # 🔥 API CALL
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )

    return response.choices[0].message.content.strip()