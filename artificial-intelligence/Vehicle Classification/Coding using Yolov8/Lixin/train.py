from ultralytics import YOLO
import os
now_path_MOP_Code = os.getcwd()
print(now_path_MOP_Code)
now_path_Vehicle_Classification = os.path.join(now_path_MOP_Code, "artificial-intelligence", "Vehicle Classification")
print(now_path_Vehicle_Classification)
now_path_Coding_using_Yolov8 = os.path.join(now_path_Vehicle_Classification, "Coding using Yolov8")
print(now_path_Coding_using_Yolov8)
now_path_MyTest = os.path.join(now_path_Coding_using_Yolov8, "Lixin")
dataset_path = os.path.join(now_path_Vehicle_Classification, "dataset")
print(os.path.join(now_path_MyTest,"yolo11n.pt"))
# Load a model
model = YOLO(os.path.join(now_path_MyTest,"yolo11n.pt"))

# Train the model
results = model.train(data=os.path.join(now_path_MyTest,"train.yaml"), epochs=100, imgsz=640)