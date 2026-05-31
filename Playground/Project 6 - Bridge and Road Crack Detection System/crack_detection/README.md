# Crack Detection — Computer Vision Module (Team Member 1)

## Overview
This module takes an infrastructure image, runs it through a trained crack segmentation model, and uploads both the original image and predicted crack mask to Supabase cloud storage.

**Pipeline:**
```
User uploads image → predict_mask() → upload to Supabase → return handoff dict
```

---

## Files
| File | Purpose |
|------|---------|
| `crack_segmentation_model.ipynb` | Training notebook — model development, evaluation, experimentation |
| `pipeline.py` | Production file — contains `run_pipeline()` for Streamlit integration |
| `.env` | put the supabase key here |
| `.gitignore` | Prevents credentials and model files from being pushed to GitHub |

---

## Setup (Team Member 2)

### Step 1 — Install dependencies
```bash
pip install supabase python-dotenv segmentation-models-pytorch albumentations torch opencv-python gdown
```

### Step 2 — Set up your .env file
`.env`, fill in the Supabase key shared by Team Member 1:
```
SUPABASE_KEY=shared_key
```
The model will be downloaded automatically from Google Drive when you first run the pipeline — no manual download needed.

### Step 3 — You are done
Run `pipeline.py` or import `run_pipeline()` in your Streamlit app.

---

## How to Use — Team Member 2 (Streamlit)

```python
import streamlit as st
from pipeline import run_pipeline

uploaded_file = st.file_uploader("Upload an infrastructure image", type=["jpg", "png"])

if uploaded_file is not None:
    # Save uploaded file temporarily
    temp_path = f"/tmp/{uploaded_file.name}"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Run the full pipeline
    result = run_pipeline(temp_path)

    # Display results
    st.image(temp_path, caption="Uploaded Image")
    st.image(result["mask_url"], caption="Predicted Crack Mask")
    st.json(result)
```

---

## How to Use — Team Member 2 (Crack Metrics)

You only need the `mask_url` from the result dict:

```python
import requests
import cv2
import numpy as np

mask_url = result["mask_url"]
response = requests.get(mask_url)
mask_array = np.frombuffer(response.content, np.uint8)
mask = cv2.imdecode(mask_array, cv2.IMREAD_GRAYSCALE)

# Run your crack metrics on mask
```

---

## What run_pipeline() Returns
```json
{
  "image_id": "abc-123-...",
  "original_url": "https://...supabase.co/.../original-images/...",
  "mask_url": "https://...supabase.co/.../crack-masks/...",
  "overlay_url": "https://...supabase.co/.../overlay_images/..."
}
```

| Key | Description |
|-----|-------------|
| `image_id` | Unique ID for this inference run |
| `original_url` | Public URL of the uploaded original image |
| `mask_url` | Public URL of the predicted crack mask |
| `overlay_url` | Public URL of the overlay image |

---

## Contact
For any issues with the pipeline or to get the Supabase key, contact Team Member 1.
