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
from urllib.parse import urljoin

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
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def preprocess_image(image_path: str) -> Optional[np.ndarray]:
    try:
        with Image.open(image_path) as img:
            img = img.convert("RGB").resize((192, 192))
            img_array = np.array(img) / 255.0
            return np.expand_dims(img_array, axis=0)
    except (UnidentifiedImageError, IOError) as e:
        logger.error(f"Invalid image file: {e}")
    return None

def safe_redirect(target: str) -> str:
    """Only allow safe internal redirects using url_for()."""
    safe_targets = ["index", "reset"]
    if target in safe_targets:
        return redirect(url_for(target))
    return redirect(url_for("index"))

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    trend = None
    feedback = None
    image_path = None
    recent_meals = get_recent_meals()

    if request.method == "POST":
        if 'file' not in request.files:
            return safe_redirect("index")

        file = request.files['file']
        if file.filename == '':
            return safe_redirect("index")

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
    """Secure reset route with safe redirect."""
    try:
        open(HISTORY_FILE, "w").close()
        return safe_redirect("index")
    except Exception as e:
        logger.error(f"Error resetting history: {e}")
        return render_template("index.html", 
                            error="Failed to reset history",
                            recent_meals=get_recent_meals())

@app.route("/health")
def health_check():
    """Health check endpoint for monitoring."""
    return {"status": "healthy"}, 200

# Safe history updater
def update_history(label_index: int) -> None:
    try:
        with open(HISTORY_FILE, "a") as f:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"{label_index},{timestamp}\n")
    except Exception as e:
        logger.error(f"Error updating history: {e}")

def get_recent_meals(max_meals: int = 7) -> List[str]:
    if not os.path.exists(HISTORY_FILE):
        return []
    
    try:
        with open(HISTORY_FILE, "r") as f:
            lines = [line.strip().split(',')[0] for line in f if line.strip()]
        return [LABEL_MAP.get(int(index), "Unknown ❓") for index in lines[-max_meals:]]
    except Exception as e:
        logger.error(f"Error reading history: {e}")
        return []

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
