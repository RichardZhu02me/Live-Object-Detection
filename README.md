# Live-Object-Detection
A python script that... detects objects live. Uses Yolov7.
Set your preferred camera in main.py->VideoCapture(i), where i is the index of the camera.
Close the program by holding 'q'. 

Current issues/ changes to be implemented:
1. heavily unoptimized: runs pretty slowly, thinking of changing the prediction to every 3ish seconds to improve playback performance or storing whatever possible to improve runtime
2. no built-in customization: camera is currently set to 1. Change it to the preferred camera and it's fine though
3. closing the program is a little awkward, because the exit button is not very helpful.
