

import sys
# sys.path.append('src')
from undistorter import Undistorter
from KalmanFilter1D import Kalman1D
from landmarks import landmarks
from head import PnPHeadPoseEstimator, HeadPoseEstimator
from face import Face
from normalization import normalize
from utils import preprocess_image
import copy
import cv2 as cv
import numpy as np
class frame_processor:
    def __init__(self, opt, cam_calib):
        self.opt = opt
        self.kalman_filters = list()
        self.face = Face()
        for point in range(2):
            # initialize kalman filters for different coordinates
            # will be used for face detection over a single object
            self.kalman_filters.append(Kalman1D(sz=10, R=0.01 ** 2))

        self.kalman_filters_landm = list()
        for point in range(68):
            self.kalman_filters_landm.append(Kalman1D(sz=100, R=0.005 ** 2))

        self.cam_calib = cam_calib
        self.undistorter = Undistorter(self.cam_calib['mtx'], self.cam_calib['dist'])
        self.landmarks_detector = landmarks()
        if opt.pose_estimator == "pnp":
            self.head_pose_estimator = PnPHeadPoseEstimator()
        elif opt.pose_estimator == "eos":
            self.head_pose_estimator = HeadPoseEstimator()
    def __call__(self, img, gaze_target = None):
        patch_type = self.opt.patch_type
        img = self.undistorter.apply(img)
        face_location = self.face.detect(img,  scale=0.25, use_max='SIZE')
        if len(face_location) > 0:
                # use kalman filter to smooth bounding box position
                # assume work with complex numbers:
            output_tracked = self.kalman_filters[0].update(face_location[0] + 1j * face_location[1])
            face_location[0], face_location[1] = np.real(output_tracked), np.imag(output_tracked)
            output_tracked = self.kalman_filters[1].update(face_location[2] + 1j * face_location[3])
            face_location[2], face_location[3] = np.real(output_tracked), np.imag(output_tracked)

                # detect facial points
            pts = self.landmarks_detector.detect(face_location, img)
                # run Kalman filter on landmarks to smooth them
            for i in range(68):
                kalman_filters_landm_complex = self.kalman_filters_landm[i].update(pts[i, 0] + 1j * pts[i, 1])
                pts[i, 0], pts[i, 1] = np.real(kalman_filters_landm_complex), np.imag(kalman_filters_landm_complex)

            fx, _, cx, _, fy, cy, _, _, _ = self.cam_calib['mtx'].flatten()
            camera_parameters = np.asarray([fx, fy, cx, cy])
            # print(img.shape, pts, camera_parameters)
            rvec, tvec, o_3d = self.head_pose_estimator(img, pts, camera_parameters)
            head_pose = (rvec, tvec)
            entry = {}
            entry["full_frame"] = img
            entry["camera_parameters"] = camera_parameters
            entry["head_pose"] = head_pose
            entry["o_3d"] = o_3d
            if gaze_target is not None:
                entry["3d_gaze_target"] = gaze_target
            normalized_entry = normalize(entry, patch_type)
            patch_img = copy.deepcopy(normalized_entry["patch"])
            normalized_entry["patch"] = preprocess_image(normalized_entry["patch"])
            ycrcb = cv.cvtColor(patch_img, cv.COLOR_BGR2YCrCb)
            ycrcb[:, :, 0] = cv.equalizeHist(ycrcb[:, :, 0])
            patch_img = cv.cvtColor(ycrcb, cv.COLOR_YCrCb2RGB)
            return True, normalized_entry, patch_img
        else:
            return False, 0, 0


