import os
import numpy as np
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from PIL import Image
from datetime import datetime

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = "static/uploaded"
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB limit
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Load models
cnn_model = load_model("health_classifier_phase3.keras")
lstm_model = load_model("habit_lstm_strict.keras")

# Config
labels = ["Healthy 🍎", "Occasional food 🟡", "Unhealthy 🍔"]
label_map = {i: label for i, label in enumerate(labels)}
history_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "user_history.txt")

# Process uploaded image
def preprocess_image(image_path):
    try:
        img = Image.open(image_path).convert("RGB")
        img = img.resize((192, 192))
        img_array = np.array(img) / 255.0
        return np.expand_dims(img_array, axis=0)
    except Exception as e:
        app.logger.error(f"Error processing image: {e}")
        return None

# Save prediction to history with timestamp
def update_history(label_index):
    try:
        with open(history_file, "a") as f:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"{label_index},{timestamp}\n")
    except Exception as e:
        app.logger.error(f"Error updating history: {e}")

# Get last 7 meal classes
def get_recent_meals(max_meals=7):
    if not os.path.exists(history_file):
        return []
    try:
        with open(history_file, "r") as f:
            lines = [line.strip().split(',')[0] for line in f if line.strip()]
        return [label_map.get(int(index), "Unknown ❓") for index in lines[-max_meals:]]
    except Exception as e:
        app.logger.error(f"Error reading history: {e}")
        return []

# Format for LSTM
def get_lstm_input(max_seq=7, num_classes=3):
    if not os.path.exists(history_file):
        lines = []
    else:
        try:
            with open(history_file, "r") as f:
                lines = [int(line.strip().split(',')[0]) for line in f.readlines()[-max_seq:]]
        except Exception as e:
            app.logger.error(f"Error reading history for LSTM: {e}")
            lines = []

    padded = [0] * (max_seq - len(lines)) + lines
    one_hot_seq = to_categorical(padded, num_classes=num_classes)
    return np.expand_dims(one_hot_seq, axis=0)

# Predict trend and feedback message
def predict_behavior():
    input_seq = get_lstm_input()
    threshold = 0.4  
    pred = lstm_model.predict(input_seq, verbose=0)[0][0]
    trend = "⚠️ Unhealthy Trend" if pred > threshold else "👍 Balanced"

    try:
        with open(history_file, "r") as f:
            recent = [int(line.strip().split(',')[0]) for line in f.readlines()[-7:]]
        unhealthy_count = recent.count(2)
        message = ""
        if unhealthy_count >= 4:
            message = "⚠️ You've had 4+ unhealthy meals recently. Try balancing with healthier choices!"
        return trend, message
    except Exception as e:
        app.logger.error(f"Error predicting behavior: {e}")
        return "Error", "Could not analyze your eating habits"

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    trend = None
    feedback = None
    image_path = None
    recent_meals = get_recent_meals()

    if request.method == "POST":
        if 'file' not in request.files:
            return redirect(request.url)
            
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
            
        if file and allowed_file(file.filename):
            try:
                safe_filename = secure_filename(file.filename)
                image_path = os.path.join(app.config['UPLOAD_FOLDER'], safe_filename)
                file.save(image_path)
                
                img_array = preprocess_image(image_path)
                if img_array is not None:
                    pred = cnn_model.predict(img_array, verbose=0)
                    label_index = np.argmax(pred[0])
                    prediction = labels[label_index]
                    update_history(label_index)
                    recent_meals = get_recent_meals()
                    trend, feedback = predict_behavior()
            except Exception as e:
                app.logger.error(f"Error processing upload: {e}")
                return render_template("index.html", 
                                     error="An error occurred while processing your image",
                                     recent_meals=recent_meals)

    return render_template("index.html",
                         prediction=prediction,
                         trend=trend,
                         image_path=image_path,
                         recent_meals=recent_meals,
                         feedback=feedback)

@app.route("/reset", methods=["POST"])
def reset():
    try:
        open(history_file, "w").close()
        recent_meals = get_recent_meals()
        return render_template("index.html", 
                             message="🧹 History reset successfully!", 
                             recent_meals=recent_meals)
    except Exception as e:
        app.logger.error(f"Error resetting history: {e}")
        return render_template("index.html", 
                             error="Failed to reset history",
                             recent_meals=get_recent_meals())

if __name__ == "__main__":
    app.run(debug=False)  # Debug should be False in production
