import sys
sys.path.append("src")
from normalization import normalize
from utils import draw_gaze
import torch
import cv2 as cv
import numpy as np 
import os 
import pickle
import h5py
import copy
from undistorter import Undistorter
from landmarks import landmarks
from head import PnPHeadPoseEstimator, HeadPoseEstimator
data_path = "/home/caixin/nas_data/data/VIPLIIGaze/supps"
dataset_path = "/home/caixin/nas_data/data/VIPLIIGaze/images"
out_path = "/home/caixin/nas_data/data/VIPLIIGaze/data"
visualize_path = "/home/caixin/nas_data/data/VIPLIIGaze/visual"
os.makedirs(out_path, exist_ok = True)
def add_key(key, value):
    to_write = out_dict
    if key not in to_write:
        to_write[key] = [value]
    else:
        to_write[key].append(value)

persons = os.listdir(dataset_path)
persons.sort()
if os.path.exists("calib_cam3_1920.pkl"):
    cam_calib = pickle.load(open("calib_cam3_1920.pkl", "rb"))
undistorter = Undistorter(cam_calib['mtx'], cam_calib['dist'])
head_pose_estimator = HeadPoseEstimator()
landmarks_detector = landmarks()
out_dict = {}
for person in persons:
    os.makedirs(os.path.join(visualize_path, person), exist_ok = True)

    out_dict = {}
    with h5py.File(os.path.join(data_path, person + ".h5"),"r") as f:
        num_entries = next(iter(f.values())).shape[0]
        for i in range(num_entries):
            img_path = '%s/%s' % (dataset_path,
                    f['file_name'][i].decode('utf-8'))
            img = cv.imread(img_path, cv.IMREAD_COLOR)
            img = undistorter.apply(img)
            entry = {}
            entry["full_frame"] = img
            entry["camera_parameters"] = f["camera_parameters"][i]

            rvec, tvec, o_3d = head_pose_estimator(img, f["landmarks"][i], f["camera_parameters"][i])
            head_pose = (rvec, tvec)
            entry["head_pose"] = head_pose
            entry["o_3d"] = o_3d
            entry["3d_gaze_target"] = f["3d_gaze_target"][i]
            normalized_entry = normalize(entry, patch_type = "face")
            add_key("face_patch", normalized_entry["patch"].astype(np.uint8))
            add_key("face_gaze", normalized_entry["normalized_gaze"])
            add_key("face_pose", normalized_entry["normalized_pose"])
            add_key("gaze_origin", normalized_entry["gaze_cam_origin"])
            add_key("gaze_target", normalized_entry["gaze_cam_target"])
            add_key("position", f["position"][i])
            add_key("cam_idx",0)

            face = copy.deepcopy(normalized_entry["patch"])
            nor_pts = landmarks_detector.detect([0,0,224,224], cv.cvtColor(face, cv.COLOR_RGB2BGR))
            add_key("nor_pts", nor_pts)
            print(f["file_name"][i])
            if i % 500 == 0:
                # print(nor_pts.shape)
                face = landmarks_detector.plot_markers(face, nor_pts.astype(int))
                vis_img = draw_gaze(face, normalized_entry["normalized_gaze"] ,color = (255,0,100))
                cv.imwrite(os.path.join(visualize_path, person, '%05d.png'%i), cv.cvtColor(vis_img, cv.COLOR_RGB2BGR))
        with h5py.File(os.path.join(out_path, person + ".h5"),"w") as g:
            for key, values in out_dict.items():
                values = np.array(values)
                print(key, values.shape)
                g.create_dataset(
                    key, data=values,
                    chunks=(
                        tuple([1] + list(values.shape[1:]))
                        if isinstance(values, np.ndarray)
                        else None
                    ),
                    compression='lzf',
                )
    # exit(0)