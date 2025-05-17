from flask import Flask, request, render_template_string
import requests

app = Flask(__name__)


API_KEY = ""

@app.route("/", methods=["GET", "POST"])
def home():
    response_text = ""
    if request.method == "POST":
        prompt = request.form["prompt"]
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={API_KEY}"
        headers = {"Content-Type": "application/json"}
        payload = {
            "contents": [
                {
                    "parts": [{"text": prompt}]
                }
            ]
        }

        try:
            res = requests.post(url, headers=headers, json=payload)
            response_text = res.json()["candidates"][0]["content"]["parts"][0]["text"]
        except Exception as e:
            response_text = f"⚠️ Error occurred: {str(e)}"

    return render_template_string('''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>🧠 Gemini Mental Health Chatbot</title>
    <style>
        body {
            background-color: #0f172a;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            color: #f1f5f9;
            margin: 0;
            padding: 30px;
        }
        h1 {
            text-align: center;
            color: #38bdf8;
        }
        .chatbox {
            background-color: #1e293b;
            padding: 20px;
            border-radius: 15px;
            box-shadow: 0 0 10px #0284c7;
            max-width: 700px;
            margin: 0 auto;
        }
        textarea {
            width: 100%;
            height: 100px;
            padding: 10px;
            border-radius: 8px;
            border: none;
            font-size: 16px;
            background-color: #334155;
            color: #f8fafc;
        }
        input[type="submit"] {
            margin-top: 15px;
            padding: 12px 25px;
            font-size: 16px;
            background-color: #38bdf8;
            color: #0f172a;
            border: none;
            border-radius: 10px;
            cursor: pointer;
            transition: background-color 0.3s;
        }
        input[type="submit"]:hover {
            background-color: #0ea5e9;
        }
        .response {
            margin-top: 30px;
            white-space: pre-wrap;
            background-color: #1e293b;
            padding: 20px;
            border-radius: 10px;
            border: 1px solid #334155;
        }
    </style>
</head>
<body>
    <h1>🧠 Mental Health Assistant (Gemini 2.0 Flash)</h1>
    <div class="chatbox">
        <form method="post">
            <textarea name="prompt" placeholder="Type your concern here...">{{ request.form.get('prompt', '') }}</textarea><br>
            <input type="submit" value="Ask">
        </form>
        {% if response %}
        <div class="response">
            <h3>Response:</h3>
            {{ response }}
        </div>
        {% endif %}
    </div>
</body>
</html>
''', response=response_text)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5005, debug=True)

