from ultralytics import YOLO
import cv2
from matplotlib import pyplot as plt

# Load trained model
model = YOLO("runs/detect/uc5_aug_train/weights/best.pt")

# Example image
img_path = "C:/Users/ramad/Downloads/dawn_yolo/images/val/0001.jpg"
results = model(img_path)

res = results[0].plot()
plt.imshow(cv2.cvtColor(res, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.show()
