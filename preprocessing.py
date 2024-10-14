import torch
import cv2
import numpy as np

def processFrame(device, frame, debug=True) :
    if debug:
        print("I am in process Frame")
    # Resize the frame to the input size expected by YOLOv7 (e.g., 640x640)
    img_size = 640
    frame_resized = cv2.resize(frame, (img_size, img_size))

    # Convert BGR (OpenCV default) to RGB as required by most models
    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)

    # Convert to a NumPy array (in case it isn't already)
    img_np = np.array(frame_rgb)

    # Normalize pixel values (from range [0, 255] to [0, 1])
    img_np = img_np / 255.0

    # Change shape from (height, width, channels) to (channels, height, width)
    img_np = np.transpose(img_np, (2, 0, 1))

    # Convert to PyTorch tensor
    img_tensor = torch.from_numpy(img_np).float()

    # Add batch dimension (1 image in this case)
    img_tensor = img_tensor.unsqueeze(0)  # Shape: (1, channels, height, width)

    # Move to the device (CPU or GPU)
    img_tensor = img_tensor.to(device)

    # Now the tensor is ready to be fed into the YOLOv7 model
    return img_tensor
