import os
import sys

# Construct the path to the yolov7 directory
yolov7_path = 'yolov7'

# Add the yolov7 directory to sys.path
sys.path.append(yolov7_path)

# Verify the path
print("YOLOv7 Path:", yolov7_path)

# Import from yolov7 after adjusting sys.path
from yolov7.models.experimental import attempt_load
