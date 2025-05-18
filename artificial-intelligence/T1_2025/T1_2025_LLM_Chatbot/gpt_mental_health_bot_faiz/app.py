from flask import Flask, render_template, request, session, redirect, url_for, send_file
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv
import os
import json
import re

app = Flask(__name__)
app.secret_key = "secret-key"

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise Exception("OpenAI API key not found!")

chat = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.3)

system_prompt = """
You are a compassionate and helpful mental health assistant.
You provide emotional support and guidance for users experiencing anxiety and depression.
Be empathetic, calm, and non-judgmental in your responses.
Never provide medical advice or make any harmful suggestions.
"""

def contains_unsafe_input(text, unsafe_terms):
    words = re.findall(r'\b\w+\b', text.lower())
    return any(term in words for term in unsafe_terms)

unsafe_inputs = [
    "kill", "suicide", "die", "hurt", "abuse",
    "end my life", "worthless", "commit suicide", "kill myself", "want to die"
]

unsafe_outputs = [
    "take your own life", "you should die", "end it all",
    "you deserve this", "i recommend suicide", "kill yourself"
]

CHAT_LOG_DIR = "chat_logs"
os.makedirs(CHAT_LOG_DIR, exist_ok=True)

def get_user_log_file():
    username = session.get("username")
    if not username:
        return None
    safe_name = "".join(c for c in username if c.isalnum() or c in "_-")
    raw_filepath = os.path.join(CHAT_LOG_DIR, f"{safe_name}.json")
    filepath = os.path.normpath(raw_filepath)

    # Ensure the filepath is within the CHAT_LOG_DIR
    if not filepath.startswith(os.path.abspath(CHAT_LOG_DIR)):
        raise Exception("Invalid file path detected.")
    return filepath

def save_chat_history():
    filepath = get_user_log_file()
    if filepath:
        data = {
            "history": session.get("history", []),
            "summary": session.get("summary"),
            "mood": session.get("mood")
        }
        with open(filepath, "w") as f:
            json.dump(data, f)

def load_chat_history():
    filepath = get_user_log_file()
    if filepath and os.path.exists(filepath):
        with open(filepath, "r") as f:
            data = json.load(f)
            if isinstance(data, list):
                session.pop("summary", None)
                session.pop("mood", None)
                return data
            elif isinstance(data, dict):
                session["summary"] = data.get("summary")
                session["mood"] = data.get("mood")
                return data.get("history", [])
    return []

@app.route("/", methods=["GET", "POST"])
def chat_page():
    if "username" not in session:
        return render_template("login.html")

    if "history" not in session:
        session["history"] = []

    if request.args.get("resume") == "yes":
        session["history"] = load_chat_history()
        session.modified = True
    elif request.args.get("resume") == "no":
        session["history"] = []
        session.pop("summary", None)
        session.pop("mood", None)
        session.modified = True

    if request.method == "POST":
        user_input = request.form["message"]

        if contains_unsafe_input(user_input, unsafe_inputs):
            session["history"].append({
                "role": "bot",
                "content": (
                    "I'm really sorry you're feeling this way. "
                    "You're not alone — please consider reaching out to a mental health professional or "
                    "contacting a helpline like Lifeline (13 11 14) or Beyond Blue (1300 22 4636) in Australia."
                )
            })
            session.modified = True
            save_chat_history()
            return redirect(url_for("chat_page"))

        session["history"].append({"role": "user", "content": user_input})
        session.modified = True

        messages = [SystemMessage(content=system_prompt)] + [
            HumanMessage(content=msg["content"]) if msg["role"] == "user"
            else AIMessage(content=msg["content"]) for msg in session["history"]
        ]

        try:
            response = chat.invoke(messages)

            if any(term in response.content.lower() for term in unsafe_outputs):
                safe_fallback = (
                    "I'm here to support you, but I cannot provide advice on this topic. "
                    "Please consider speaking to a qualified mental health professional for guidance."
                )
                session["history"].append({"role": "bot", "content": safe_fallback})
            else:
                session["history"].append({"role": "bot", "content": response.content})

                if len(session["history"]) >= 6 and len(session["history"]) % 2 == 0:
                    summary_prompt = (
                        "Summarize the conversation between user and bot. "
                        "Then, provide a mood label for the user. "
                        "Choose ONLY ONE from: happy, sad, anxious. "
                        "Format as JSON like:\n"
                        "{\n  \"summary\": \"...summary here...\",\n  \"mood\": \"happy\"\n}"
                    )
                    recent_msgs = session["history"][-6:]
                    summary_msgs = [SystemMessage(content=summary_prompt)] + [
                        HumanMessage(content=m["content"]) if m["role"] == "user"
                        else AIMessage(content=m["content"]) for m in recent_msgs
                    ]

                    try:
                        summary_response = chat.invoke(summary_msgs)
                        raw = summary_response.content.strip()

                        try:
                            parsed = json.loads(raw)
                            session["summary"] = parsed.get("summary")
                            session["mood"] = parsed.get("mood")
                        except Exception as e:
                            print("⚠️ Failed to parse JSON from GPT:", e)
                            session["summary"] = raw
                            session["mood"] = None

                        session.modified = True
                    except Exception as e:
                        print("Summary generation failed:", e)

            session.modified = True
            save_chat_history()

        except Exception as e:
            session["history"].append({"role": "bot", "content": f"Error: {e}"})
            session.modified = True
            save_chat_history()

        return redirect(url_for("chat_page"))

    return render_template(
        "chat.html",
        history=session.get("history", []),
        summary=session.get("summary"),
        mood=session.get("mood")
    )

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username")
        if username:
            session["username"] = username

            try:
                filepath = get_user_log_file()
            except:
                return "❌ Invalid username."

            if os.path.exists(filepath):
                return redirect(url_for("resume_prompt"))
            else:
                session["history"] = []
                session.pop("summary", None)
                session.pop("mood", None)
                return redirect(url_for("chat_page"))

    return render_template("login.html")

@app.route("/reset")
def reset_chat():
    session.pop("history", None)
    session.pop("summary", None)
    session.pop("mood", None)

    try:
        filepath = get_user_log_file()
        if filepath and os.path.exists(filepath):
            os.remove(filepath)
            print("✅ File deleted:", filepath)
    except Exception as e:
        print("❌ Error deleting chat log:", e)

    return redirect(url_for("chat_page"))

@app.route("/summarize", methods=["POST"])
def summarize():
    if "history" not in session or len(session["history"]) < 4:
        return redirect(url_for("chat_page"))

    summary_prompt = (
        "Summarize the conversation between user and bot. "
        "Then, provide a mood label for the user. "
        "Choose ONLY ONE from: happy, sad, anxious. "
        "Format as JSON like:\n"
        "{\n  \"summary\": \"...summary here...\",\n  \"mood\": \"happy\"\n}"
    )
    recent_msgs = session["history"][-6:]

    summary_msgs = [SystemMessage(content=summary_prompt)] + [
        HumanMessage(content=m["content"]) if m["role"] == "user"
        else AIMessage(content=m["content"]) for m in recent_msgs
    ]

    try:
        summary_response = chat.invoke(summary_msgs)
        raw = summary_response.content.strip()

        try:
            parsed = json.loads(raw)
            session["summary"] = parsed.get("summary")
            session["mood"] = parsed.get("mood")
        except Exception as e:
            print("⚠️ Failed to parse JSON from GPT:", e)
            session["summary"] = raw
            session["mood"] = None

        session.modified = True
        save_chat_history()
    except Exception as e:
        print("Manual summary failed:", e)

    return redirect(url_for("chat_page"))

@app.route("/resume")
def resume_prompt():
    return render_template("resume.html")

@app.route("/bubbles")
def bubbles():
    return render_template("bubbles.html")

@app.route("/download")
def download_chat():
    try:
        filepath = get_user_log_file()
        if filepath and os.path.exists(filepath):
            return send_file(filepath, as_attachment=True)
    except Exception as e:
        print("❌ Download error:", e)
    return "No file available."

if __name__ == "__main__":
    debug_mode = os.getenv("FLASK_DEBUG", "false").lower() == "true"
    app.run(debug=debug_mode, port=5020)
