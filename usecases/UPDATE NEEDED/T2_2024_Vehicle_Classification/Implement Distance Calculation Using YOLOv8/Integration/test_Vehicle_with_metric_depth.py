from ultralytics import YOLO
import os
import cv2
import torch
import matplotlib
import numpy as np
import argparse
from ultralytics.utils.plotting import Annotator
from metric_depth.depth_anything_v2.dpt import DepthAnythingV2




DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

parser = argparse.ArgumentParser(description='Depth')
    
parser.add_argument('--img-path', type=str,default='metric_depth/test/pic/BDouble_10_1.jpg')
parser.add_argument('--input-size', type=int, default=518)
parser.add_argument('--outdir', type=str, default='metric_depth/test/res')
parser.add_argument('--encoder', type=str, default='vits', choices=['vits', 'vitb', 'vitl', 'vitg'])
parser.add_argument('--grayscale', dest='grayscale', action='store_true', help='do not apply colorful palette')

args = parser.parse_args()

model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}

encoder = 'vits' # or 'vits', 'vitb', 'vitg'

max_depth=80

model = DepthAnythingV2(**{**model_configs[encoder], 'max_depth': max_depth})
model.load_state_dict(torch.load('metric_depth/models/depth_metric_vits.pth', map_location='cuda:0'))

model = model.to(DEVICE).eval()


raw_img = cv2.imread(args.img_path)
raw_depth = model.infer_image(raw_img) 


cmap = matplotlib.colormaps.get_cmap('Spectral_r')
depth=raw_depth
depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
depth = depth.astype(np.uint8)

if args.grayscale:
    depth = np.repeat(depth[..., np.newaxis], 3, axis=-1)
else:
    depth = (cmap(depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)

# cv2.imwrite('test.png', depth)

# Load a model
# model = YOLO("yolov8s.yaml")  # build a new model from scratch
model_yolo = YOLO("metric_depth/models/yolo_best.pt")  # load a pretrained model (recommended for training)

results = model_yolo.predict(source=args.img_path,save=False,device=1,conf=0.4)

annotator = Annotator(raw_img)
annotator_depth = Annotator(depth)

font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.5
color = (0,0,255)  
thickness = 1

for box in results[0].boxes:
            
    b = box.xyxy[0]  # get box coordinates in (left, top, right, bottom) format
    c = box.cls
    annotator.box_label(b, model_yolo.names[int(c)])
    annotator_depth.box_label(b, model_yolo.names[int(c)])

img=annotator_depth.result()

for box in results[0].boxes:
            
    b = box.xyxy[0]  # get box coordinates in (left, top, right, bottom) format
    c = box.cls
    
    x1=int(b[0])
    y1=int(b[1])
    x2=int(b[2])
    y2=int(b[3])

    box_depth=raw_depth[x1:x2,y1:y2]
    max_dis='{:.2f}'.format(box_depth.max())
    min_dis='{:.2f}'.format(box_depth.min())
    
    text_size, _ = cv2.getTextSize('Max: '+str(max_dis)+'m', font, font_scale, thickness)
    position = (x1, y1+20)  
    position2 = (x1+text_size[0]+10, y1+20)  

   
    cv2.putText(img, 'Max: '+str(max_dis)+'m', position, font, font_scale, color, thickness)
    cv2.putText(img, 'Min: '+str(min_dis)+'m', position2, font, font_scale, color, thickness)
        
split_region = np.ones((raw_img.shape[0], 50, 3), dtype=np.uint8) * 255
combined_result = cv2.hconcat([annotator.result(), split_region, img])
cv2.imwrite(os.path.join(args.outdir, os.path.splitext(os.path.basename(args.img_path))[0] + '.png'), combined_result)


