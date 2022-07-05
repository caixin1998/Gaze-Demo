import cv2 as cv
import numpy as np
import os 
from shutil import copyfile
id = 3
cam_idx = 6
videos_path = "/home/caixin/nas_data/VIPLIIGaze/calibration/3"
output_path = "/home/caixin/nas_data/data/VIPLIIGaze/images"
# videodata = skvideo.io.vread(os.path.join(video_path, "cam0.mp4"))
cut_num = 25

videos = os.listdir(videos_path)
videos.sort()
for video in videos:
    person = video.split("_")[0]
    video_path = os.path.join(videos_path, video)
    labels = np.loadtxt(os.path.join(video_path, "calib_target.txt"))
    time0 = np.loadtxt(os.path.join(video_path, "time0.txt"))
    times = np.loadtxt(os.path.join(video_path, "times.txt"))
    assert len(labels) == len(times)
    save_labels = np.repeat(labels, cut_num, axis = 0)
    os.makedirs(os.path.join(output_path, person, "cam0", video), exist_ok = True)
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
            cv.imwrite("%s/%s/%s/%s/%05d.png"%(output_path, person, "cam0", video, i), frame)
            i = i + 1
            # print("%s/%s/%05d.png"%(output_path,video, i))
        ret, frame = cap.read()
        idx += 1
    assert i == len(save_labels)
    print(os.path.join(output_path, person, "cam0", video))
    cap.release()

    np.savetxt(os.path.join("/home/caixin/nas_data/data/VIPLIIGaze/images",person, "cam0", video, "labels.txt"), save_labels)
    copyfile(os.path.join(videos_path, video, "position.txt"), os.path.join(output_path, person, "cam0", video,"position.txt"))
    copyfile("/home/caixin/tnm-opencv/data/%s/cam%s/opt.txt"%(id, cam_idx), os.path.join(output_path, person, "cam0", video,"ext.txt"))


