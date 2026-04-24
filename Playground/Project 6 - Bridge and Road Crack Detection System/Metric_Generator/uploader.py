import requests
import os


url = "http://127.0.0.1:5000/api/uploadImage"

image_path = "masks/crack_mask (55).jpg"

with open(image_path, "rb") as f:
    response = requests.post(url, files={"file": (os.path.basename(image_path), f, "image/jpeg")})
    
print(response.json())