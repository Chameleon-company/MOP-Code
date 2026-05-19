from ultralytics import YOLO
import os

# Load a model
# model = YOLO("yolov8s.yaml")  # build a new model from scratch
model = YOLO(r"/applications/deakin/teamproject/MOP-Code/runs/detect/train25/weights/best.pt")  # load a pretrained model (recommended for training)

# Use the model
results = model.predict(source=r"/applications/deakin/teamproject/MOP-Code/artificial-intelligence/Vehicle Classification/Coding using Yolov8/Testing performance/test1.avi",device='cpu',stream=True,conf=0.7)

# Process results list
index=0
for result in results:
    index=index+1
    boxes = result.boxes  # Boxes object for bounding box outputs
    # masks = result.masks  # Masks object for segmentation masks outputs
    # keypoints = result.keypoints  # Keypoints object for pose outputs
    # probs = result.probs  # Probs object for classification outputs
    # obb = result.obb  # Oriented boxes object for OBB outputs
    # # result.show()  # display to screen
    path=r'/applications/deakin/teamproject/MOP-Code/artificial-intelligence/Vehicle Classification/Coding using Yolov8/Lixin/test_res_imgs'
    if not os.path.exists(path):
        os.makedirs(path)
    if boxes.shape[0]!=0:
        result.save(filename=os.path.join(path,str(index)+'_result.jpg'))  # save to disk
  # predict on an image
# path = model.export(format="onnx")  # export the model to ONNX format