# index.py
import os

from flask import Flask, jsonify, request

app = Flask(__name__)


@app.route("/")
def index():
    return jsonify({'status': 'running'})


@app.route("/order", methods=["POST"])
def create_order():
    data = request.get_json()
    print('Request Data: ' + str(data))
    return jsonify({'msg': 'Order Created Successfully'})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
