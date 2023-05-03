
import time
import sys
sys.path.append("src")
import cv2 as cv
import numpy as np
from head import PnPHeadPoseEstimator, HeadPoseEstimator

from face import Face
from landmarks_media import landmarks
import eos
import pickle
import os   
from normalization import normalize


def get_normalization_image(img_path):
    img = cv.imread(img_path)
    face = Face()
    head_pose_estimator = HeadPoseEstimator()
 

    landmarks_detector = landmarks()
 
    pts = landmarks_detector.detect(img)
    if pts is None:
        return
    pts_int = np.array(pts, dtype = np.int32)
    landmark_img = landmarks_detector.plot_markers(img, pts_int, color=(0, 255, 0), radius=3, drawline=False)
    cv.imwrite("data/landmark_img.png", landmark_img)

    
    if os.path.exists("intrinsic/538/calib_cam3.pkl"):
        cam_calib = pickle.load(open("intrinsic/538/calib_cam3.pkl", "rb"))
        rvec, tvec, o_3d = head_pose_estimator(img, pts, cam_calib["mtx"])
        head_pose = (rvec, tvec)
        print(head_pose)
        entry = {}
        entry["full_frame"] = img
        entry["camera_parameters"] = cam_calib["mtx"]
        entry["head_pose"] = head_pose
        entry["o_3d"] = o_3d
        normalized_entry = normalize(entry, patch_type = "face")
        cv.imwrite("data/media_normalized_face%s.png"%id,cv.cvtColor(normalized_entry["patch"].astype(np.uint8), cv.COLOR_RGB2BGR))
        
for id in range(1, 7):
    img_path = "data/demo%s.png"%id
    get_normalization_image(img_path)