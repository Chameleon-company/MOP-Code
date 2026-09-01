from ultralytics import YOLO
import os
import cv2
import torch
import matplotlib
import numpy as np
import argparse
from ultralytics.utils.plotting import Annotator
from depth.dpt import DepthAnythingV2




DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

parser = argparse.ArgumentParser(description='Depth')
    
parser.add_argument('--img-path', type=str,default='depth/test/pic/BDouble_10_1.jpg')
parser.add_argument('--input-size', type=int, default=518)
parser.add_argument('--outdir', type=str, default='depth/test/res')
parser.add_argument('--encoder', type=str, default='vits', choices=['vits', 'vitb', 'vitl', 'vitg'])
parser.add_argument('--pred-only', dest='pred_only', action='store_true', help='only display the prediction')
parser.add_argument('--grayscale', dest='grayscale', action='store_true', help='do not apply colorful palette')

args = parser.parse_args()

model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}

encoder = 'vits' # or 'vits', 'vitb', 'vitg'

model = DepthAnythingV2(**model_configs[encoder])
model.load_state_dict(torch.load('depth/models/depth_vits.pth', map_location='cuda:0'))
model = model.to(DEVICE).eval()

cmap = matplotlib.colormaps.get_cmap('Spectral_r')

filename=args.img_path
raw_img = cv2.imread(filename)
depth = model.infer_image(raw_img) 

depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
depth = depth.astype(np.uint8)

if args.grayscale:
    depth = np.repeat(depth[..., np.newaxis], 3, axis=-1)
else:
    depth = (cmap(depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)

# Load a model
# model = YOLO("yolov8s.yaml")  # build a new model from scratch
model_yolo = YOLO("depth/models/yolo_best.pt")  # load a pretrained model (recommended for training)

results = model_yolo.predict(source="depth/test/pic/BDouble_10_1.jpg",save=False,device=1,conf=0.4)

annotator = Annotator(raw_img)
annotator_depth = Annotator(depth)

for box in results[0].boxes:
            
    b = box.xyxy[0]  # get box coordinates in (left, top, right, bottom) format
    c = box.cls
    annotator.box_label(b, model_yolo.names[int(c)])
    annotator_depth.box_label(b, model_yolo.names[int(c)])
        

split_region = np.ones((raw_img.shape[0], 50, 3), dtype=np.uint8) * 255
combined_result = cv2.hconcat([annotator.result(), split_region, annotator_depth.result()])
cv2.imwrite(os.path.join(args.outdir, os.path.splitext(os.path.basename(filename))[0] + '.png'), combined_result)


