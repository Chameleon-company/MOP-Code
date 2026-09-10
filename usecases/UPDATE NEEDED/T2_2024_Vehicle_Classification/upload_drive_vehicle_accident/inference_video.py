import torch
from ultralytics import YOLO

# Check if CUDA is available and print the GPU name
print("CUDA available:", torch.cuda.is_available())  # Should return True if CUDA is enabled
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))  # Should return the name of your GPU

# Load the trained model weights
model_path = "/Users/nguyenbui/Documents/upload_drive_vehicle_accident/best.pt"
model = YOLO(model_path)  # Load the trained model

# Move the model to GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

# Set the video path and output path
video_path = "/Users/nguyenbui/Documents/upload_drive_vehicle_accident/NTTA Traffic Cam Video Released Of Deadly Wrong-Way Crash On PGBT In Richardson.mp4"
output_path = "/Users/nguyenbui/Documents/upload_drive_vehicle_accident/predictions/processed_video.mp4"

# Perform detection on the video and save the output
results = model.predict(source=video_path, device=device, save=True, save_dir="/Users/nguyenbui/Documents/upload_drive_vehicle_accident/predictions",conf=0.5)

# Print a message to confirm that the video has been saved
print(f"Prediction video saved to: {output_path}")
