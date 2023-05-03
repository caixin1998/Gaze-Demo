import numpy as np
import mediapipe as mp
import cv2

# unfinished code;maybe useful in the future
from face_geometry import get_metric_landmarks, PCF, canonical_metric_landmarks, procrustes_landmark_basis

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh


class MediaHeadPoseEstimator(object):

    def __init__(self) -> None:
        points_idx = [33,263,61,291,199]
        points_idx = points_idx + [key for (key,val) in procrustes_landmark_basis]
        points_idx = list(set(points_idx))
        points_idx.sort()
        self.points_idx = points_idx
        self.face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

    def detect_landmarks(self, frame, visualize=False):
        frame_height, frame_width, channels = frame.shape
        results = self.face_mesh.process(frame)
        multi_face_landmarks = results.multi_face_landmarks

        if multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            landmarks = np.array([(lm.x,lm.y,lm.z) for lm in face_landmarks.landmark])
            landmarks =  landmarks.T

            image_points = landmarks[0:2, self.points_idx].T* np.array([frame_width, frame_height])[None,:]
            return landmarks, image_points
        else: 
            return None, None

    def __call__(self, frame, landmarks, camera_matrix, visualize=False):
        frame_height, frame_width, channels = frame.shape

        pcf = PCF(near=1,far=10000,frame_height=frame_height,frame_width=frame_width,fy=camera_matrix[1,1])

        # print(idx)
        metric_landmarks, pose_transform_mat = get_metric_landmarks(landmarks.copy(), pcf)
        model_points = metric_landmarks[self.points_idx, 0:3]
        print(model_points)
        image_points = landmarks[0:2, self.points_idx].T * np.array([frame_width, frame_height])[None,:]
        success, rvec, tvec = cv2.solvePnP(model_points, image_points, camera_matrix, None, flags=cv2.cv2.SOLVEPNP_ITERATIVE)
        transform = np.asmatrix(np.eye(4))
        transform[:3, :3] = cv2.Rodrigues(rvec)[0]
        transform[:3, 3] = tvec

        # landmark_points_68 = [162,234,93,58,172,136,149,148,152,377,378,365,397,288,323,454,389,71,63,105,66,107,336,296,334,293,301,168,197,5,4,75,97,2,326,305,33,160,158,133,153,144,362,385,387,263,373,380,61,39,37,0,267,269,291,405,314,17,84,181,78,82,13,312,308,317,14,87]

        # Define 3D gaze origin coordinates
        # Ref. iBUG(start from 1) -to- 468 indices
        #       37   =>  33  # right eye outer-corner (1)
        #       40   =>  133  # right eye inner-corner (5)
        #       43   =>  362  # left eye inner-corner (8)
        #       46   =>  263  # left eye outer-corner (2)
        #       31   =>  4  # nose-tip (3)
        #       34   =>  2  # nose-lip junction

        o_3d = {}
        o_3d["reyes"] = np.mean([metric_landmarks[33], metric_landmarks[133]], axis=0)
        o_3d["leyes"] = np.mean([metric_landmarks[362], metric_landmarks[263]], axis=0)
        o_3d_eye_center = np.mean([o_3d["leyes"], o_3d["reyes"]], axis=0)
        o_3d_nose = np.mean([metric_landmarks[4], metric_landmarks[2]], axis=0)
        o_3d["face"] = np.mean([o_3d_eye_center, o_3d_nose], axis=0)
        o_3d["eyes"] = np.mean([metric_landmarks[133], metric_landmarks[362]], axis=0)

        for k, v in o_3d.items():
            o_3d[k] = np.asarray(np.matmul(transform, np.asmatrix([*v, 1.0]).reshape(-1, 1)))[:3, :]

        return rvec, tvec, o_3d

         # _, rotation_vector, translation_vector, inliers = cv2.solvePnPRansac(model_points, image_points, camera_matrix, dist_coeff)

            # (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 25.0)]), rotation_vector, translation_vector, camera_matrix, None)

            # for ii in self.points_idx: # range(landmarks.shape[1]):
            #     pos = np.array((frame_width*landmarks[0, ii], frame_height*landmarks[1, ii])).astype(np.int32)
            #     frame = cv2.circle(frame, tuple(pos), 1, (0, 255, 255), -1)

            # p1 = (int(image_points[0][0]), int(image_points[0][1]))
            # p2 = ( int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))

            # frame = cv2.line(frame, p1, p2, (255,0,0), 2)
