import os
import logging
import numpy as np
from typing import List, Tuple, Optional
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
from flask_wtf.csrf import CSRFProtect
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from PIL import Image, UnidentifiedImageError
from datetime import datetime

# Initialize Flask application
app = Flask(__name__)
app.config.update(
    UPLOAD_FOLDER="static/uploaded",
    MAX_CONTENT_LENGTH=16 * 1024 * 1024,  # 16MB limit
    ALLOWED_EXTENSIONS={'png', 'jpg', 'jpeg', 'gif'},
    SECRET_KEY=os.getenv('FLASK_SECRET_KEY', 'dev-secret-key-please-change')  # Change in production
)

# Initialize CSRF protection
csrf = CSRFProtect(app)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load ML models
try:
    cnn_model = load_model("health_classifier_phase3.keras")
    lstm_model = load_model("habit_lstm_strict.keras")
except Exception as e:
    logger.error(f"Failed to load models: {e}")
    raise

# Application constants
LABELS = ["Healthy 🍎", "Occasional food 🟡", "Unhealthy 🍔"]
LABEL_MAP = {i: label for i, label in enumerate(LABELS)}
HISTORY_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "user_history.txt")

def allowed_file(filename: str) -> bool:
    """Check if the file has an allowed extension."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def preprocess_image(image_path: str) -> Optional[np.ndarray]:
    """Process uploaded image for model prediction."""
    try:
        with Image.open(image_path) as img:
            img = img.convert("RGB").resize((192, 192))
            img_array = np.array(img) / 255.0
            return np.expand_dims(img_array, axis=0)
    except (UnidentifiedImageError, IOError) as e:
        logger.error(f"Invalid image file: {e}")
    except Exception as e:
        logger.error(f"Error processing image: {e}")
    return None

def update_history(label_index: int) -> None:
    """Record prediction in history file with timestamp."""
    try:
        with open(HISTORY_FILE, "a") as f:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"{label_index},{timestamp}\n")
    except Exception as e:
        logger.error(f"Error updating history: {e}")

def get_recent_meals(max_meals: int = 7) -> List[str]:
    """Retrieve recent meal classifications from history."""
    if not os.path.exists(HISTORY_FILE):
        return []
    
    try:
        with open(HISTORY_FILE, "r") as f:
            lines = [line.strip().split(',')[0] for line in f if line.strip()]
        return [LABEL_MAP.get(int(index), "Unknown ❓") for index in lines[-max_meals:]]
    except Exception as e:
        logger.error(f"Error reading history: {e}")
        return []

def get_lstm_input(max_seq: int = 7, num_classes: int = 3) -> np.ndarray:
    """Prepare input data for LSTM model."""
    if not os.path.exists(HISTORY_FILE):
        lines = []
    else:
        try:
            with open(HISTORY_FILE, "r") as f:
                lines = [int(line.strip().split(',')[0]) for line in f.readlines()[-max_seq:]]
        except Exception as e:
            logger.error(f"Error reading history for LSTM: {e}")
            lines = []

    padded = [0] * (max_seq - len(lines)) + lines
    one_hot_seq = to_categorical(padded, num_classes=num_classes)
    return np.expand_dims(one_hot_seq, axis=0)

def predict_behavior() -> Tuple[str, str]:
    """Analyze eating habits and provide feedback."""
    input_seq = get_lstm_input()
    threshold = 0.4
    try:
        pred = lstm_model.predict(input_seq, verbose=0)[0][0]
        trend = "⚠️ Unhealthy Trend" if pred > threshold else "👍 Balanced"

        with open(HISTORY_FILE, "r") as f:
            recent = [int(line.strip().split(',')[0]) for line in f.readlines()[-7:]]
        
        unhealthy_count = recent.count(2)
        message = "⚠️ You've had 4+ unhealthy meals recently. Try balancing with healthier choices!" \
                 if unhealthy_count >= 4 else ""
        return trend, message
    except Exception as e:
        logger.error(f"Error predicting behavior: {e}")
        return "Error", "Could not analyze your eating habits"

@app.route("/", methods=["GET", "POST"])
def index():
    """Main application route handling file uploads and predictions."""
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
                    prediction = LABELS[label_index]
                    update_history(label_index)
                    recent_meals = get_recent_meals()
                    trend, feedback = predict_behavior()
            except Exception as e:
                logger.error(f"Error processing upload: {e}")
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
    """Reset the meal history."""
    try:
        open(HISTORY_FILE, "w").close()
        recent_meals = get_recent_meals()
        return render_template("index.html", 
                            message="🧹 History reset successfully!", 
                            recent_meals=recent_meals)
    except Exception as e:
        logger.error(f"Error resetting history: {e}")
        return render_template("index.html", 
                            error="Failed to reset history",
                            recent_meals=get_recent_meals())

@app.route("/health")
def health_check():
    """Health check endpoint for monitoring."""
    return {"status": "healthy"}, 200

if __name__ == "__main__":
    # Determine environment
    env = os.getenv('FLASK_ENV', 'development')
    
    if env == 'production':
        logger.info("Starting production server")
        from waitress import serve
        serve(app, host="0.0.0.0", port=5000)
    else:
        debug_mode = os.getenv('FLASK_DEBUG', 'False').lower() in ('true', '1', 't')
        logger.info(f"Starting {'debug' if debug_mode else 'development'} server")
        app.run(host="0.0.0.0", port=5000, debug=debug_mode)
