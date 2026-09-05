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
import tempfile

load_dotenv(Path(__file__).parent.parent / '.env.local')

# ── Supabase Setup ────────────────────────────────────────────────────────────
SUPABASE_URL     = os.environ.get("SUPABASE_URL")
SUPABASE_KEY     = os.environ.get("SUPABASE_PUBLISHABLE_DEFAULT_KEY") or os.environ.get("SUPABASE_KEY")
BUCKET_ORIGINALS = "original-images"
BUCKET_MASKS     = "crack-masks"
BUCKET_OVERLAYS  = "overlay_images"

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


def overlay_crack_on_image(image_path: str, mask_path: str, output_path: str) -> str:
    """
    Overlay predicted crack mask on original image in red.
    Returns path to the overlay image.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"[Overlay] Could not read image: {image_path}")

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"[Overlay] Could not read mask: {mask_path}")

    mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

    overlay = image.copy()
    overlay[mask > 127] = [0, 0, 255]  # red overlay on crack pixels

    result = cv2.addWeighted(image, 0.7, overlay, 0.3, 0)

    # output_dir is already created by run_pipeline() — only create if called standalone
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    success = cv2.imwrite(output_path, result)
    if not success:
        raise IOError(f"[Overlay] cv2.imwrite failed — could not write to: {output_path}")

    print(f"[Overlay] Saved → {output_path}")
    return output_path


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


def push_inference_result(original_path: str, mask_path: str, image_id: str) -> dict:
    """
    Upload original image and predicted mask to Supabase.
    Returns handoff dict for Team Member 2 (Crack Metrics).
    """
    orig = Path(original_path)
    mask = Path(mask_path)

    orig_remote = f"{image_id}_original{orig.suffix}"
    mask_remote = f"{image_id}_mask{mask.suffix}"

    print(f"[Supabase] Uploading original → {orig_remote}")
    original_url = upload_to_bucket(original_path, BUCKET_ORIGINALS, orig_remote)
    print(f"[Supabase] ✓ Original uploaded")

    print(f"[Supabase] Uploading mask    → {mask_remote}")
    mask_url = upload_to_bucket(mask_path, BUCKET_MASKS, mask_remote)
    print(f"[Supabase] ✓ Mask uploaded")

    return {
        "image_id":     image_id,
        "original_url": original_url,
        "mask_url":     mask_url
    }


def run_pipeline(input_image_path: str, output_dir: str = None) -> dict:
    """
    Full pipeline entry point.
    Input:  path to a user-uploaded infrastructure image
    Output: handoff dict with image_id, original_url, mask_url, overlay_url

    Usage in Streamlit (Team Member 4):
        result = run_pipeline("/tmp/uploaded_image.jpg")
        mask_url     = result["mask_url"]      # pass to Team Member 2
        overlay_url  = result["overlay_url"]   # display in dashboard
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

        # Use system temp folder if no output_dir provided (Windows compatible)
        if output_dir is None:
            output_dir = os.path.join(tempfile.gettempdir(), "crack_outputs")
        os.makedirs(output_dir, exist_ok=True)

        image_id            = str(uuid.uuid4())
        output_mask_path    = os.path.join(output_dir, f"{image_id}_mask.png")
        output_overlay_path = os.path.join(output_dir, f"{image_id}_overlay.png")

        # Step 1: Generate mask
        predict_mask(input_image_path, output_mask_path)

        # Step 2: Generate overlay
        overlay_crack_on_image(input_image_path, output_mask_path, output_overlay_path)

        # Step 3: Upload all to Supabase
        result = push_inference_result(input_image_path, output_mask_path, image_id)
        overlay_url = upload_to_bucket(
            output_overlay_path,
            BUCKET_OVERLAYS,
            f"{image_id}_overlay.png"
        )
        result["overlay_url"] = overlay_url

        # Step 4: Save handoff JSON
        with open(os.path.join(output_dir, f"{image_id}_handoff.json"), "w") as f:
            json.dump(result, f, indent=2)

        print(f"\n[Pipeline] Handoff ready:")
        print(f"           image_id     → {image_id}")
        print(f"           original_url → {result['original_url']}")
        print(f"           mask_url     → {result['mask_url']}")
        print(f"           overlay_url  → {overlay_url}")

        return result

    except FileNotFoundError as e:
        print(f"[Pipeline Error] File not found: {e}")
        raise
    except ValueError as e:
        print(f"[Pipeline Error] Invalid image: {e}")
        raise
    except Exception as e:
        import traceback
        print(f"[Pipeline Error] Something went wrong: {e}")
        traceback.print_exc()
        raise