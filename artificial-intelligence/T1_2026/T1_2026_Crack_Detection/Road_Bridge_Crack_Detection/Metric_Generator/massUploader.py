import requests
import os
from pathlib import Path
import time



url = "http://127.0.0.1:5000/api/uploadImage"


folder = Path("masks")
    
reports = []

start = time.time()

for img_path in folder.iterdir():
    if img_path.suffix.lower() in (".jpg", ".jpeg", ".png"):
        with open(img_path, "rb") as f:
            response = requests.post(url, files={"file": (os.path.basename(img_path), f, "image/jpeg")})
            print(response.json())
            
elapsed = time.time() - start
print(f"Elapsed: {elapsed:.2f}s")