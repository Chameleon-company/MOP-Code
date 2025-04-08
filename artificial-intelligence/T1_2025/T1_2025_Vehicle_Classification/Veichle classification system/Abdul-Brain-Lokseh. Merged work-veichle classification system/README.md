# Vehicle Classification Ensemble System

This project implements a vehicle classification system using an ensemble of three different deep learning models:

1.  **PyTorch EfficientNet-B0:** A pre-trained model from `timm` library, fine-tuned on the vehicle dataset.
2.  **TensorFlow Custom CNN ("Brian's Model"):** A custom Convolutional Neural Network built with Keras Sequential API. *(Note: This model was included primarily for collaborative and demonstration purposes to integrate multiple frameworks).*
3.  **Ultralytics YOLOv8-Classification:** A pre-trained YOLOv8 classification model (`yolov8n-cls.pt`), fine-tuned on the vehicle dataset.

The system trains these models individually (currently set for 1 epoch for testing) and then combines their predictions using a weighted averaging ensemble strategy. Performance is evaluated on validation and test sets.

## Features

*   Trains models using PyTorch, TensorFlow/Keras, and Ultralytics frameworks.
*   Implements weighted ensembling for potentially improved performance.
*   Includes Test Time Augmentation (TTA - horizontal flip) during evaluation.
*   Generates classification reports and confusion matrices.
*   Visualizes sample predictions.
*   Configurable settings via `config.py`.
*   Code structured into logical Python modules.

## Setup

### Prerequisites

*   Python 3.9+
*   `pip` package manager

### Installation

1.  **Clone the repository (if applicable):**
    ```bash
    # git clone <your-repo-url>
    # cd <your-repo-directory>
    ```

2.  **Create a Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    # Activate the environment
    # Windows:
    .\venv\Scripts\activate
    # macOS/Linux:
    source venv/bin/activate
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *Note:* Installing PyTorch and TensorFlow, especially with GPU support, might require specific commands based on your OS and CUDA version. Refer to their official websites if the `requirements.txt` installation fails for these packages.

4.  **Dataset Structure:**
    *   Create a directory named `datasets` in the main project folder.
    *   Inside `datasets`, create `train`, `val`, and `test` subdirectories.
    *   Populate each subdirectory with folders named according to your classes (e.g., `class_1`, `class_2`, ...). Place the corresponding images inside these class folders.
    *   Example: `datasets/train/class_1/image1.jpg`

5.  **Dataset Configuration File (`vehicle_data.yaml`):**
    *   Create a file named `vehicle_data.yaml` in the main project directory (same level as `main.py`).
    *   Paste the following content into it, **adjusting `nc` and `names`** if your dataset differs:

      ```yaml
      # Base path (relative to where main.py is run)
      path: datasets

      # Dataset paths (relative to base path)
      train: train
      val: val
      test: test # Optional if test set isn't used by YOLO training directly

      # Number of classes
      nc: 13 # MODIFY if different

      # Class names (ensure order matches directory sorting)
      names: ['class_1', 'class_2', 'class_3', 'class_4', 'class_5', 'class_6', 'class_7', 'class_8', 'class_9', 'class_10', 'class_11', 'class_12', 'class_13'] # MODIFY if different
      ```

## Configuration

Key parameters can be adjusted in `config.py`:

*   `DATASET_PATH`: Path to your dataset root.
*   `*_EPOCHS`: Number of epochs for each model (currently set to 1).
*   `*_MODEL_WEIGHT`: Weights for each model in the ensemble.
*   `DEVICE`: Automatically detects CUDA GPU or uses CPU.
*   `NUM_WORKERS`: Set to `0` for CPU training/debugging, increase for GPU if stable.
*   `VERBOSE`, `PRINT_MODEL_SUMMARIES`: Control output detail.

## Running the System

Ensure your virtual environment is activated. Run the main script from the project's root directory:

```bash
python main.py
