import torch

import cv2 as cv
import numpy as np
import os 
import sys
sys.path.append("src")
import pickle
import h5py
from monitor import monitor
from face import Face
from landmarks import landmarks
from undistorter import Undistorter

landmarks_detector = landmarks()
face = Face()
# videos_path = "/home/caixin/nas_data/VIPLIIGaze/calibration/3"
img_path = "/home/caixin/nas_data/data/VIPLIIGaze/images"
output_path = "/home/caixin/nas_data/data/VIPLIIGaze/supps"
os.makedirs(output_path, exist_ok = True)
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
undistorter = Undistorter(cam_calib['mtx'], cam_calib['dist'])
for person in persons:
    # if os.path.isfile(os.path.join(output_path, person + ".h5")):
    #     continue
    cams = os.listdir(os.path.join(img_path,person))
    for cam in cams:
        videos = os.listdir(os.path.join(img_path,person,cam))
    for video in videos:
        imgs_path = os.path.join(img_path, person , cam, video)
        labels = np.loadtxt(os.path.join(imgs_path, "labels.txt"))
        ext = os.path.join(imgs_path, "ext.txt")
        position = np.loadtxt(os.path.join(imgs_path, "position.txt"))
        if labels.shape[1] == 2:
            mon = monitor(ext)

        person = video.split("_")[0]
        imgs = os.listdir(imgs_path)

        for i, img in enumerate(imgs):
            if not img.endswith("png"):
                break
            file_name = os.path.join(person, cam, video, img)
            if labels.shape[1] == 2:
                label = mon.monitor_to_camera(labels[i][0], labels[i][1])  
            else:
                label = labels[i]  
            print(file_name)
            frame = cv.imread(os.path.join(img_path, file_name))
            frame = undistorter.apply(frame)
            face_location = face.detect(frame,scale=0.25, use_max='SIZE')
            if len(face_location) == 4:
               add_key(person, "face_valid", 1) 
               landmarks = landmarks_detector.detect(face_location, frame)
            else:
                add_key(person, "face_valid", 0)
                landmarks = np.zeros(68,2)
            add_key(person, "landmarks", landmarks)
            add_key(person, "3d_gaze_target", label)
            add_key(person, "file_name", file_name)
            add_key(person, "position", position)
            add_key(person, "camera_parameters", camera_parameters)
            add_key(person, "distortion_parameters", distortion_parameters)
           
    

    
    with h5py.File(os.path.join(output_path, person + ".h5"), "w") as f:     
        for key, values in out_dict[person].items():
            values = np.array(values)
            print(key, values.shape)
            if key == "file_name":
                ds = f.create_dataset(
                    key, values.shape,
                dtype = h5py.special_dtype(vlen=str)
                )
                ds[:] = values
            else:
                f.create_dataset(
                    key, data=values,
                    chunks=(
                        tuple([1] + list(values.shape[1:]))
                        if isinstance(values, np.ndarray)
                        else None
                    ),
                    compression='lzf',
                )

            # print(key, np.array(out_dict[person][key]).shape)
        



