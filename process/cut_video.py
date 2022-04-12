import cv2 as cv
import numpy as np
import os 
videos_path = "/home/caixin/nas_data/VIPLIIGAZE/calibration/3"
output_path = "/home/caixin/nas_data/data/VIPLIIGAZE/images"
# videodata = skvideo.io.vread(os.path.join(video_path, "cam0.mp4"))
cut_num = 25
for video in os.listdir(videos_path):
    labels = np.loadtxt(os.path.join(video_path, "calib_target.txt"))

    save_labels = np.repeat(labels, cut_num, axis = 0)
    video_path = os.path.join(videos_path, video)
    time0 = np.loadtxt(os.path.join(video_path, "time0.txt"))
    times = np.loadtxt(os.path.join(video_path, "times.txt"))
    assert len(labels) == len(times)
    frame_idxs = []
    for i in times:
        idx = np.argmin(abs(time0 - i))
        frame_idxs += list(range(idx - cut_num + 1, idx + 1))   
    cap = cv.VideoCapture(os.path.join(video_path, "cam0.mp4"))
    i = 0
    idx = 0
    ret, frame = cap.read()
    while ret and i < len(frame_idxs):
        if idx == frame_idxs[i]:
            cv.imwrite("output_path/%s/%d.png"%i,person,frame)
            i = i + 1
            print(i)
        ret, frame = cap.read()
        idx += 1
    assert idx - 1 == len(save_labels)
    
    cap.release()

