import torch
import cv2
import numpy as np
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2

class CrackSegmentor:
    def __init__(self, weights_path: str, threshold: float = 0.5):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.threshold = threshold

        self.model = smp.Unet(
            encoder_name="resnet34",
            encoder_weights=None,
            in_channels=3,
            classes=1,
        )
        self.model.load_state_dict(
            torch.load(weights_path, map_location=self.device)
        )
        self.model.to(self.device)
        self.model.eval()

        self._transform = A.Compose([
            A.Resize(224, 224),
            A.Normalize(),
            ToTensorV2(),
        ])

    def predict(self, image_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns (original_rgb, prob_map, binary_mask)."""
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original = image.copy()

        tensor = self._transform(image=image)["image"].unsqueeze(0).float().to(self.device)

        with torch.no_grad():
            prob = torch.sigmoid(self.model(tensor)).squeeze().cpu().numpy()

        mask = (prob > self.threshold).astype(np.uint8)
        return original, prob, mask