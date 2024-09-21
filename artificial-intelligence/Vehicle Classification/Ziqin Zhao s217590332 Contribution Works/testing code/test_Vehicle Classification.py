from ultralytics import YOLO
import os

# Load a model
# model = YOLO("yolov8s.yaml")  # build a new model from scratch
model = YOLO("metric_depth/models/yolo_best.pt")  # load a pretrained model (recommended for training)

# Use the model
results = model.predict(source="tests/test_source/test1.mkv",save=False,device='cpu',stream=True,conf=0.7)

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
    path='tests/test_res'
    if not os.path.exists(path):
        os.makedirs(path)
    if boxes.shape[0]!=0:
        result.save(filename=os.path.join(path,str(index)+'_result.jpg'))  # save to disk
  # predict on an image
# path = model.export(format="onnx")  # export the model to ONNX format