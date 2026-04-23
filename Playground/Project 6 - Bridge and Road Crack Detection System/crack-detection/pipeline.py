import os
import cv2
import json
import uuid
import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pathlib import Path
from supabase import create_client, Client
import segmentation_models_pytorch as smp
from dotenv import load_dotenv
import gdown

load_dotenv()

# ── Supabase Setup ────────────────────────────────────────────────────────────
SUPABASE_URL     = "https://lpazsslmpirwtfywndou.supabase.co"
SUPABASE_KEY     = os.environ["SUPABASE_KEY"]
BUCKET_ORIGINALS = "original-images"
BUCKET_MASKS     = "crack-masks"

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ── Device ────────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Load Model ────────────────────────────────────────────────────────────────
MODEL_PATH = os.environ.get("MODEL_PATH", "model1_unet_resnet34.pth")

if not os.path.exists(MODEL_PATH):
    print("Downloading model from Google Drive...")
    gdown.download(
        "https://drive.google.com/uc?id=1R_efjnaTWWnOrg27qYum3jv3Xv4zr3qS",
        MODEL_PATH,
        quiet=False
    )
    print("Model downloaded!")

model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights=None,
    in_channels=3,
    classes=1
)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()
print(f"Model loaded from: {MODEL_PATH}")
print(f"Running on: {device}")

# ── Inference Transform ───────────────────────────────────────────────────────
infer_transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(),
    ToTensorV2()
])


# ── Core Functions ────────────────────────────────────────────────────────────
def predict_mask(image_path: str, output_mask_path: str) -> str:
    """
    Run inference on a single image and save the predicted binary crack mask.
    Returns the path to the saved mask.
    """
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    original_h, original_w = image_bgr.shape[:2]
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    transformed = infer_transform(image=image_rgb)
    tensor = transformed["image"].unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(tensor)
        pred   = (torch.sigmoid(output) > 0.5).float()

    mask = pred.squeeze().cpu().numpy()
    mask = (mask * 255).astype(np.uint8)
    mask = cv2.resize(mask, (original_w, original_h), interpolation=cv2.INTER_NEAREST)

    os.makedirs(os.path.dirname(output_mask_path), exist_ok=True)
    cv2.imwrite(output_mask_path, mask)
    print(f"[Inference] Mask saved → {output_mask_path}")
    return output_mask_path


def upload_to_bucket(local_path: str, bucket: str, remote_name: str) -> str:
    """Upload a file to a Supabase Storage bucket. Returns the public URL."""
    local_path   = Path(local_path)
    ext          = local_path.suffix.lower()
    content_type = "image/png" if ext == ".png" else "image/jpeg"

    with open(local_path, "rb") as f:
        file_bytes = f.read()

    supabase.storage.from_(bucket).upload(
        path=remote_name,
        file=file_bytes,
        file_options={"content-type": content_type, "upsert": "true"}
    )
    return supabase.storage.from_(bucket).get_public_url(remote_name)


def run_pipeline(input_image_path: str, output_dir: str = "/tmp/outputs") -> dict:
    """
    Full pipeline entry point.
    Input:  path to a user-uploaded infrastructure image
    Output: handoff dict with image_id, original_url, mask_url

    Usage in Streamlit:
        result = run_pipeline("/tmp/uploaded_image.jpg")
        mask_url     = result["mask_url"]
        original_url = result["original_url"]
        image_id     = result["image_id"]
    """
    try:
        # Validate image exists
        if not os.path.exists(input_image_path):
            raise FileNotFoundError(f"Image not found: {input_image_path}")

        # Validate image is readable
        test = cv2.imread(input_image_path)
        if test is None:
            raise ValueError(f"Image could not be read, check format: {input_image_path}")

        os.makedirs(output_dir, exist_ok=True)
        image_id         = str(uuid.uuid4())
        output_mask_path = os.path.join(output_dir, f"{image_id}_mask.png")

        # Step 1: Generate mask
        predict_mask(input_image_path, output_mask_path)

        # Step 2: Upload both to Supabase
        original_url = upload_to_bucket(
            input_image_path,
            BUCKET_ORIGINALS,
            f"{image_id}_original{Path(input_image_path).suffix}"
        )
        mask_url = upload_to_bucket(
            output_mask_path,
            BUCKET_MASKS,
            f"{image_id}_mask.png"
        )

        result = {
            "image_id":     image_id,
            "original_url": original_url,
            "mask_url":     mask_url
        }

        # Step 3: Save handoff JSON
        with open(os.path.join(output_dir, f"{image_id}_handoff.json"), "w") as f:
            json.dump(result, f, indent=2)

        print(f"\n[Pipeline] Handoff ready:")
        print(f"           image_id     → {image_id}")
        print(f"           original_url → {original_url}")
        print(f"           mask_url     → {mask_url}")

        return result

    except FileNotFoundError as e:
        print(f"[Pipeline Error] File not found: {e}")
        raise
    except ValueError as e:
        print(f"[Pipeline Error] Invalid image: {e}")
        raise
    except Exception as e:
        print(f"[Pipeline Error] Something went wrong: {e}")
        raise
