from sahi import AutoDetectionModel
import cv2
from ultralytics import YOLO
import os
from utils import get_model_with_sahi_cuda,get_model_with_sahi_cpu
from detect import get_sahi_prediction,get_yolo_prediction,merge_img_horizen,test_performance
now_path_MOP_Code = os.getcwd()                                 
vehicle_path = os.path.join(now_path_MOP_Code, "artificial-intelligence", "Vehicle Classification")
# # yolo11s
# yolov11_path = os.path.join(vehicle_path, "Coding using Yolov8","Lixin","yolo11s.pt")
# our model
yolov11_path = os.path.join(vehicle_path, "Coding using Yolov8","Lixin","yolov11","best.pt")
# build model
model,yolo_model = get_model_with_sahi_cuda(yolov11_path)
# test_image
input_path = os.path.join(vehicle_path, "Coding using Yolov8","Lixin","sahi","input")
out_path = os.path.join(vehicle_path, "Coding using Yolov8","Lixin","sahi","output")

test1_path = os.path.join(input_path,"small-vehicles1.jpeg")
test2_path = os.path.join(input_path,"terrain2.png")
out1 = os.path.join(out_path, "small-vehicles1.jpeg")
out2 = os.path.join(out_path, "terrain2.png")
# test_performance(test1_path,out1,model)
test_performance(test1_path,out1,model,yolo_model)
test_performance(test2_path,out2,model,yolo_model,128,128,0.2,0.2)