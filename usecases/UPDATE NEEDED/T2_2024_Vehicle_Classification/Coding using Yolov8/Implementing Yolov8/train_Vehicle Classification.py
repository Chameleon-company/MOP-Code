from ultralytics import YOLO

# Load a model
model = YOLO("yolov8s.yaml")  # build a new model from scratch
# model = YOLO("best.pt") # load a pretrained model (recommended for training)

# Use the model
model.train(data="/home/Ziqinzhao/work/yolov8/ultralytics-main/ultralytics/cfg/datasets/Vehicle Classification.yaml", batch=16, epochs=200,device='0',resume=False)  # train the model
metrics = model.val()  # evaluate model performance on the validation set
results = model("/home/Ziqinzhao/work/yolov8/datasets/764/test/BDouble_10_1.jpg") 
for result in results:
    boxes = result.boxes  # Boxes object for bounding box outputs
    result.save(filename="result.jpg")  # predict on an image
path = model.export(format="onnx")  # export the model to ONNX format