import os
import cv2
from vision import drawframe, load

# open camera
# draw a frame
# process frame
# show frame

# main function: captures an image of the camera, then draws on it
def main() : 
    # Open the default camera, default camera here is 1
    cam = cv2.VideoCapture(1)
    
    if not cam.isOpened():
        print("Error: Camera not opened or not available.")
        return
    
    else:
       print("Camera opened successfully.")    
    
    device, model = load()
    while True:
        ret, frame = cam.read()
        
        processed_frame = drawframe(device, model, frame, debug=False)
        # Display the captured frame
        cv2.imshow('window',processed_frame)

        # Press 'q' to exit the loop
        if cv2.waitKey(1) == ord('q'):
            break

    # Release the capture and writer objects
    cam.release()
    # out.release()
    cv2.destroyAllWindows()
    
main()