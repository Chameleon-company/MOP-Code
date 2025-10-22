# index.py
import os

from flask import Flask, jsonify, request, send_file

app = Flask(__name__)


@app.route("/")
def index():
    return jsonify({'status': 'running'})


@app.route("/order", methods=["POST"])
def create_order():
    data = request.get_json()
    print('Request Data: ' + str(data))
    return jsonify({'msg': 'Order Created Successfully'})


@app.route("/download", methods=["GET"])
def download_test():
    from data import DataStorageFactory
    repo = DataStorageFactory().create('google')
    filename = 'parkingsensor.csv'
    stream = repo.get(f"parkingsensor/{filename}")
    return send_file(stream, download_name=filename)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
