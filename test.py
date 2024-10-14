# Tests the program on the downloaded images.
import os
import cv2
from vision import drawframe, load


# main function: captures an image of the camera, then draws on it
print("I am in main")
# Open the default camera, default camera here is 1
# cam = cv2.VideoCapture()
device, model = load()

# ret, frame = cam.read()

frame = cv2.imread('images/dogcat.jpg')

processed_frame = drawframe(device, model, frame)
# Display the captured frame
cv2.imshow('window',processed_frame)
cv2.waitKey(0)
if cv2.waitKey(1) == ord('q'):
    cv2.destroyAllWindows()