from flask import Flask, render_template, request, jsonify
import requests

app = Flask(__name__, static_url_path='/static')

RASA_URL = "http://localhost:5005/webhooks/rest/webhook"

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/send_message", methods=["POST"])
def send_message():
    user_message = request.form["message"]
    response = requests.post(RASA_URL, json={"sender": "user", "message": user_message})
    return jsonify(response.json())

if __name__ == "__main__":
    app.run(debug=True)
    
