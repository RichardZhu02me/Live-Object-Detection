import os
import cv2
from vision import drawframe
for filename in uploaded.keys():
    # Save the uploaded file to the local file system
    with open(filename, 'wb') as f:
        f.write(uploaded[filename])

    # Create the 'data' directory if it doesn't exist
    if not os.path.exists('data'):
        os.makedirs('data')

    # Move uploaded file to the correct directory
    os.rename(filename, f"data/{filename}")
    


# main function: captures an image of the camera, then draws on it
def main() : 
    cam = cv2.VideoCapture(0) 
    while True:
        drawframe(capture(cam))
        key = cv2.waitKey(1)
        if key == 'q':
            break

    cam.release()
    cv2.destroyAllWindows()
    
    
    
def capture(cam) :
    
    check, frame = cam.read()

    cv2.imshow('video', frame)