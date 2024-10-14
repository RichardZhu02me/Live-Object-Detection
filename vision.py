import sys
sys.path.append('yolov7')
import torch
# from pathlib import Path
from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords, set_logging
from utils.datasets import LoadImages
from utils.torch_utils import select_device
import cv2
import numpy as np
import matplotlib.pyplot as plt
from preprocessing import processFrame

# Initialize
# set_logging()
def load() :
    device = select_device('')  # Automatically selects GPU if available, otherwise uses CPU
    model = attempt_load('yolov7/weights/yolov7.pt', map_location=device)  # Load YOLOv7 model
    return device, model

# # Load an image
# img_path = 'images/household.webp'  # Replace with your image file name
def drawimage(img_path, device, model) :
    dataset = LoadImages(img_path, img_size=640)

    # Process the image
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.float() / 255.0
        if len(img.shape) == 3:
            img = img.unsqueeze(0)

        # Inference
        pred = model(img, augment=False)[0]
        pred = non_max_suppression(pred, 0.4, 0.5)

        # Process detections
        for det in pred:
            if len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0s.shape).round()

                # Draw boxes
                for *xyxy, conf, cls in det:
                    label = f'{model.names[int(cls)]} {conf:.2f}'
                    im0s = cv2.rectangle(im0s, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 255, 0), 2)
                    im0s = cv2.putText(im0s, label, (int(xyxy[0]), int(xyxy[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Save or display results
        # result_img_path = 'result.jpg'
        # cv2.imwrite(result_img_path, im0s)

        # Display the image using cv2_imshow, passing the image data (im0s) instead of the file path
        cv2.imshow('results',im0s)
        
        
def drawframe(device, model, frame, debug=True) :
    if debug:
        print("I am in vision")
    img_tensor = processFrame(device, frame, debug)
    if img_tensor.ndimension() == 3:
        img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension


    # Inference
    pred = model(img_tensor, augment=False)[0]
    pred = non_max_suppression(pred, 0.4, 0.5)

    # Process detections
    for det in pred:
        if len(det):
            det[:, :4] = scale_coords(img_tensor.shape[2:], det[:, :4], frame.shape).round()

            # Draw boxes
            for *xyxy, conf, cls in det:
                label = f'{model.names[int(cls)]} {conf:.2f}'
                frame = cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 255, 0), 2)
                frame = cv2.putText(frame, label, (int(xyxy[0]), int(xyxy[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return frame