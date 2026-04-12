from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Layer
import tensorflow as tf
import numpy as np
from PIL import Image
import os

class Cast(Layer):
    def call(self, inputs):
        return tf.identity(inputs)

app = Flask(__name__)
model = load_model("health_classifier_phase2.keras")

UPLOAD_FOLDER = "static/uploaded"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def preprocess_image(image_path):
    img = Image.open(image_path).convert("RGB")
    img = img.resize((192, 192))
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    image_path = None

    if request.method == "POST":
        file = request.files["file"]
        if file:
            image_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(image_path)

            img_array = preprocess_image(image_path)
            pred = model.predict(img_array)
            prediction = "Healthy🟢" if np.argmax(pred[0]) == 0 else "Unhealthy🔴"

    return render_template("index.html", prediction=prediction, image_path=image_path)

if __name__ == "__main__":
    app.run(debug=True)
