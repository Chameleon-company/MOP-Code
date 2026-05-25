import json

def load_data():
    with open("data/sample_data.json") as f:
        return json.load(f)