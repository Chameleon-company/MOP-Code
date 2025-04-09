# Vehicle Classification Ensemble (PyTorch + YOLO)

This project trains and evaluates an ensemble model combining PyTorch EfficientNet-B0 and YOLOv8-Large-CLS for vehicle classification. It incorporates techniques like Focal Loss, MixUp, and Weighted Sampling to handle class imbalance.

## Setup

1.  **Clone the repository:**
    ```bash
    git clone [Your Repository URL]
    cd vehicle-classification-ensemble
    ```

2.  **Create/Activate Environment (Recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # Linux/macOS
    # venv\Scripts\activate  # Windows
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Dataset:**
    *   Create a `datasets` folder in the project root.
    *   Inside `datasets`, create `train`, `val`, and `test` subfolders.
    *   Inside each of these, create subfolders named `class_1`, `class_2`, ..., `class_13` containing the corresponding images (.jpg, .png, etc.).

## How to Run

Execute the main training and evaluation script:

```bash
python train_ensemble.py
