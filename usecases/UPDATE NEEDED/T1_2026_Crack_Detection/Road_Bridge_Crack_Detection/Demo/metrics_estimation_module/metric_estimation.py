import sys
import json
from PIL import Image
from .crackAnalyser import generateMetricReport
from pathlib import Path
import numpy as np

class MetricEstimator:
    def estimate(self, mask: np.ndarray, image_id: str = "image")  -> dict:
        """Run metric estimation on a single mask image. Returns the report dict."""
        pil_mask = Image.fromarray((mask * 255).astype(np.uint8)).convert("RGB")
        pil_mask.format = "PNG"  # satisfy the format check in generateMetricReport
        return generateMetricReport(pil_mask, image_id)
