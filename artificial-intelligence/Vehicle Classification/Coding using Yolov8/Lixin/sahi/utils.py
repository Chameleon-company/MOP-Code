from sahi.predict import get_sliced_prediction,get_prediction,PredictionResult
from sahi import AutoDetectionModel
import cv2
from ultralytics import YOLO
from ultralytics.engine.results import Results

import os
class BOX:
    """
    A class to represent a bounding box with coordinates and category.
    Attributes:
    ----------
    minx : float
        The minimum x-coordinate of the bounding box.
    miny : float
        The minimum y-coordinate of the bounding box.
    maxx : float
        The maximum x-coordinate of the bounding box.
    maxy : float
        The maximum y-coordinate of the bounding box.
    category : str
        The category label of the bounding box.
    Methods:
    -------
    __str__():
        Returns a string representation of the bounding box coordinates.
    __repr__():
        Returns a string representation of the bounding box coordinates.
    get_center():
        Calculates and returns the center coordinates of the bounding box.
    merge_label():
        Merges multiple labels into the most common label and updates the category.
    """
    def __init__(self,minx,miny,maxx,maxy,category):
        self.minx = minx
        self.miny = miny
        self.maxx = maxx
        self.maxy = maxy
        self.category = category
    
    def __str__(self):
        return f"minx: {self.minx}, miny: {self.miny}, maxx: {self.maxx}, maxy: {self.maxy}"
    def __repr__(self):
        return f"minx: {self.minx}, miny: {self.miny}, maxx: {self.maxx}, maxy: {self.maxy}"
    def get_center(self):
        return ((self.minx+self.maxx)/2,(self.miny+self.maxy)/2)
    def merge_label(self):
        labels = self.category.split('_')
        dic = {}
        for label in labels:
            if label in dic:
                dic[label] += 1
            else:
                dic[label] = 1
        most_common = max(dic,key=dic.get)
        self.category = most_common
        return most_common
        
## extract BOXes from yolo result
def get_BOXes_from_yolo_result(result:Results):
    boxes = result.boxes
    cls_indexs = boxes.cls
    names = result.names
    for i in range(len(boxes)):
        box = boxes[i].xyxy
        cls_index = cls_indexs[i]
        name = names[int(cls_index)]
        yield BOX(box[0][0],box[0][1],box[0][2],box[0][3],name)

## extract BOXes from sahi result
def get_BOXes_from_sahi_result(result:PredictionResult):
    for object_prediction in result.object_prediction_list:
        bbox = object_prediction.bbox
        category = object_prediction.category.name
        yield BOX(bbox.minx,bbox.miny,bbox.maxx,bbox.maxy,category)

def get_model_with_sahi_cuda(model_path):
    detection_model = AutoDetectionModel.from_pretrained(
        model_type='yolov8',
        model_path=model_path,
        confidence_threshold=0.6,
        device='cuda:0'
    )
    yolo_model = YOLO(model_path)
    return detection_model,yolo_model

def get_model_with_sahi_cpu(model_path):
    detection_model = AutoDetectionModel.from_pretrained(
        model_type='yolov8',
        model_path=model_path,
        confidence_threshold=0.6,
        device='cpu'
    )
    yolo_model = YOLO(model_path)
    return detection_model,yolo_model


def draw_BOX(image, box, color=(0, 0, 255), thickness=1):
    box.merge_label()
    x1, y1, x2, y2 = int(box.minx), int(box.miny), int(box.maxx), int(box.maxy)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
    cv2.putText(image, box.category, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    return image

def draw_bbox_sahi(image, BoudingBox, color=(0, 0, 255), thickness=1):
    bbox = BoudingBox.bbox
    cato = BoudingBox.category.name
    box = BOX(bbox.minx,bbox.miny,bbox.maxx,bbox.maxy,cato)
    return draw_BOX(image, box, color, thickness)



def merge_img_horizen(img1,img2):
    return cv2.hconcat([img1, img2])

def calculate_iou(bbox1, bbox2):
    x1 = max(bbox1.minx, bbox2.minx)
    y1 = max(bbox1.miny, bbox2.miny)
    x2 = min(bbox1.maxx, bbox2.maxx)
    y2 = min(bbox1.maxy, bbox2.maxy)
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (bbox1.maxx - bbox1.minx) * (bbox1.maxy - bbox1.miny)
    area2 = (bbox2.maxx - bbox2.minx) * (bbox2.maxy - bbox2.miny)
    union = area1 + area2 - intersection
    iou = intersection / area1
    return iou
def merge_overlapping_boxes(BOX_list:list):
    merged_boxes = []
    for box in BOX_list:
        merged = False
        for merged_box in merged_boxes:
            iou = calculate_iou(box, merged_box)
            print(iou)
            if iou > 0.:  # Adjust the threshold as needed
                merged_box.minx = min(box.minx, merged_box.minx)
                merged_box.miny = min(box.miny, merged_box.miny)
                merged_box.maxx = max(box.maxx, merged_box.maxx)
                merged_box.maxy = max(box.maxy, merged_box.maxy)
                merged = True
                merged_box.category = box.category+'_'+merged_box.category
                break
        if not merged:
            merged_boxes.append(box)
    return merged_boxes

