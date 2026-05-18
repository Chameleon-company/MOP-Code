from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import json
import os
import requests

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
json_path = os.path.join(BASE_DIR, "recycling_data_clean.json")
with open(json_path, "r", encoding="utf-8") as f:
    recycling_data = json.load(f)

def get_relevant_items(message):
    message_lower = message.lower()
    relevant = []
    for entry in recycling_data:
        if entry["item"].lower() in message_lower:
            relevant.append(entry)
        else:
            for alias in entry.get("aliases", []):
                if alias.lower() in message_lower:
                    relevant.append(entry)
    if not relevant:
        relevant = recycling_data[:3]
    return relevant

def build_prompt(message, relevant_items):
    details_list = []
    for item in relevant_items:
        detail = f"- Item: {item['item']}\n"
        detail += f"  Instructions: {item['instructions']}\n"
        detail += f"  Bin Type: {item.get('bin_type', 'N/A')}\n"
        if item.get("bin_color"):
            detail += f"  Bin Color: {item['bin_color']}\n"
        if item.get("drop_off_locations"):
            locs = []
            for loc in item["drop_off_locations"]:
                loc_str = loc.get("name", "Unknown location")
                if loc.get("website"):
                    loc_str += f" ({loc['website']})"
                locs.append(loc_str)
            detail += f"  Drop-off Locations: {', '.join(locs)}\n"
        if item.get("website"):
            detail += f"  Website: {item['website']}\n"
        if item.get("extra_notes"):
            detail += f"  Extra Notes: {item['extra_notes']}\n"
        details_list.append(detail)
    details_text = "\n".join(details_list)
    prompt = f"""You are a helpful recycling assistant in Victoria, Australia.
Use the following information about items to answer the user's question in a friendly and helpful way.

Information:
{details_text}

User question: {message}"""
    return prompt

def ask_mistral(prompt):
    API_KEY = "Ol1lWKKBKfFKSXpN5OWCgR7weAWPOTiO"
    url = "https://api.mistral.ai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "mistral-tiny",
        "messages": [{"role": "user", "content": prompt}]
    }
    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print("Mistral API error:", e)
        return "Sorry, I could not generate an answer right now."


@app.route("/ask", methods=["POST"])
def ask():
    question = request.json.get("question")
    if not question:
        return jsonify({"error": "No question provided"}), 400
    relevant_items = get_relevant_items(question)
    prompt = build_prompt(question, relevant_items)
    answer = ask_mistral(prompt)
    return jsonify({"answer": answer})

if __name__ == "__main__":
    app.run(port=5000, debug=True)
