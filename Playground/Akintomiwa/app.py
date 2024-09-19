from flask import Flask, request, jsonify
import pickle
import numpy as np
import os

# Initializing Flask app
app = Flask(__name__)

# Load the saved RandomForest model
with open('random_forest_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

@app.route('/predict', methods=['POST'])
def predict():
    # Getting the input data from the POST request
    data = request.get_json()  # expects data in JSON format
    input_features = np.array([data['input']])  # Convert input to the required format

    # Make prediction using the loaded model
    prediction = model.predict(input_features)

    # Return the prediction as a JSON response
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    debug_mode = os.getenv('FLASK_DEBUG', 'False').lower() in ['true', '1', 't']
    app.run(debug=debug_mode)
