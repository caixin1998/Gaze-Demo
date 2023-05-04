
import time
import sys
sys.path.append("src")
import cv2 as cv
import numpy as np
from head import PnPHeadPoseEstimator, HeadPoseEstimator
from face import Face
from landmarks import landmarks
import eos
import pickle
import os   
from normalization import normalize
def viewport_matrix(w,h):
    viewport = np.array([0,h,w,-h])
    
    # scale
    S = np.identity(4,dtype=np.float32)
    S[0][0] *= viewport[2]/2
    S[1][1] *= viewport[3]/2
    S[2][2] *= 0.5
    
    # translate
    T = np.identity(4,dtype=np.float32)
    T[3][0] = viewport[0] + (viewport[2]/2)
    T[3][1] = viewport[1] + (viewport[3]/2)
    T[3][2] = 0.5
    return S@T


def get_normalization_image(img_path):
    img = cv.imread(img_path)
    face = Face()
    head_pose_estimator = HeadPoseEstimator()
    face_location = face.detect(img,  scale=0.25, use_max='SIZE')
    print(face_location)
    face_location = np.array(face_location, dtype = np.int32)
    origin_img = img.copy()
    #crop face use face_location
    # face_img = img[face_location[1]:face_location[3], face_location[0]:face_location[2]]
    face_img = cv.rectangle(origin_img,face_location[:2], face_location[2:], (0, 0, 255), 4)
    #save face_img
    # cv.imwrite("data/face_img.png", face_img)

    landmarks_detector = landmarks()
    pts = landmarks_detector.detect(face_location, img)
    # pts = np.array(pts, dtype = np.int32)
    # print(pts)
    # landmark_img = landmarks_detector.plot_markers(img, pts, color=(0, 255, 0), radius=3, drawline=False)
    # cv.imwrite("data/landmark_img.png", landmark_img[face_location[1]:face_location[3], face_location[0]:face_location[2]])

    # eos_mesh, eos_pose, eos_shape_coeffs, eos_blendshape_coeffs = head_pose_estimator.mesh_fit(img, pts)
    # # image = cv.cvtColor(img, cv.COLOR_BGR2BGRA, 4)
    # # isomap = eos.render.extract_texture(eos_mesh, eos_pose, image)
    # # isomap = cv.cvtColor(isomap, cv.COLOR_BGRA2BGR)
    # # eos_img = np.transpose(isomap, [1, 0, 2])
    # h,w = img.shape[:2]
    # vm = viewport_matrix(w,h)
    # canvas = img.copy()
    # p = eos_pose.get_projection()
    # mv = eos_pose.get_modelview()
    # fm = vm@p@mv
    # for i in eos_mesh.vertices:
    #     tmp = fm@np.append(i,1)
    #     # disregard z and draw 2d pt
    #     x,y = (int(w/2+tmp[0]),int(h/2+tmp[1]))
    #     cv.circle(canvas,(x,y),2,(0,255,0),thickness=-1)
    # cv.imwrite("data/eos_img.png", canvas[face_location[1]:face_location[3], face_location[0]:face_location[2]])

    if os.path.exists("intrinsic/538/calib_cam3.pkl"):
        cam_calib = pickle.load(open("intrinsic/538/calib_cam3.pkl", "rb"))
        rvec, tvec, o_3d = head_pose_estimator(img, pts, cam_calib["mtx"])
        head_pose = (rvec, tvec)
        entry = {}
        entry["full_frame"] = img
        entry["camera_parameters"] = cam_calib["mtx"]
        entry["head_pose"] = head_pose
        entry["o_3d"] = o_3d
        normalized_entry = normalize(entry, patch_type = "face")
        cv.imwrite("data/normalized_face%s.png"%id,cv.cvtColor(normalized_entry["patch"].astype(np.uint8), cv.COLOR_RGB2BGR))
tic = time.time()
for id in range(1, 7):
    img_path = "data/demo%s.png"%id
    get_normalization_image(img_path)
print(time.time() - tic)