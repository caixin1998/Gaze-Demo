import torch

import cv2 as cv
import numpy as np
import os 
import sys
sys.path.append("../src")
sys.path.append("..")

import monitor
from face import Face()
from landmarks import landmarks
landmarks_detector = landmarks()
# videos_path = "/home/caixin/nas_data/VIPLIIGaze/calibration/3"
img_path = "/home/caixin/nas_data/data/VIPLIIGaze/images"
# videodata = skvideo.io.vread(os.path.join(video_path, "cam0.mp4"))
cut_num = 25
if os.path.exists("calib_cam3_1920.pkl"):
    cam_calib = pickle.load(open("calib_cam3_1920.pkl", "rb"))
camera_parameters = cam_calib["mtx"]
distortion_parameters = cam_calib["dist"]

persons = os.listdir(img_path)
persons.sort()

out_dict = {}
for person in persons:
    out_dict[person] = {}

def add_key(person, key, value):
    to_write = out_dict[person]
    if key not in to_write:
        to_write[key] = [value]
    else:
        to_write[key].append(value)

for person in persons:
    cams = os.listdir(os.path.join(img_path,person))
    for cam in cams:
        videos = os.listdir(os.path.join(img_path,person,cam))
    for video in videos:
        imgs_path = os.path.join(img_path, person , cam, video)
        labels = np.loadtxt(os.path.join(imgs_path, "labels.txt"))
        ext = os.path.join(imgs_path, "ext.txt")
        position = os.path.join(imgs_path, "position.txt")
        if labels.shape[1] == 2:
            mon = monitor(ext)

        person = video.split("_")[0]
        imgs = os.listdir(imgs_path)
        for i, img in enumerate(imgs):
            file_name = os.path.join(person, cam, video, img)
            if labels.shape[1] == 2:
                label = mon.monitor_to_camera(labels[i])  
            else:
                label = labels[i]  
            print(file_name)
            frame = cv.imread(file_name)
            face_location = face.detect(frame,scale=0.25, use_max='SIZE')
            if len(face_location) == 4:
               add_key(person, "face_valid", 1) 
               landmarks = landmarks_detector.detect(face_location, frame)
            else:
                add_key(person, "face_valid", 0)
                landmarks = np.zeros(68,2)
            add_key(person, "3d_gaze_target", label)
            add_key(person, "file_name", file_name)
            add_key(person, "position", position)
            add_key(person, "camera_parameters", camera_parameters)
            add_key(person, "distortion_parameters", distortion_parameters)
            add_key(person, "landmarks", landmarks)

        for key in out_dict[person]:
            print(key, np.array(out_dict[person][key]).shape)




