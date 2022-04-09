import cv2 as cv
import numpy as np
import os 
import skvideo.io  
video_path = "calibration/2/jiabei_10+30"
# videodata = skvideo.io.vread(os.path.join(video_path, "cam0.mp4"))
time0 = np.loadtxt(os.path.join(video_path, "time0.txt"))
times = np.loadtxt(os.path.join(video_path, "times.txt"))
frame_idxs = []
for i in times:
    idx = np.argmin(abs(time0 - i))
    frame_idxs += list(range(idx - 29, idx + 1))    
cap = cv.VideoCapture(os.path.join(video_path, "cam0.mp4"))
i = 0
idx = 0
ret, frame = cap.read()
while ret and i < len(frame_idxs):
    if idx == frame_idxs[i]:
        cv.imwrite("temp/%d.png"%i,frame)
        i = i + 1
        print(i)
    ret, frame = cap.read()
    idx += 1
 
cap.release()

