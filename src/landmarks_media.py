import numpy as np
import mediapipe as mp
import cv2



mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh


class landmarks:

    def __init__(self) -> None:
        
        self.landmark_points_68 = [162,234,93,58,172,136,149,148,152,377,378,365,397,288,323,454,389,71,63,105,66,107,336,296,334,293,301,168,197,5,4,75,97,2,326,305,33,160,158,133,153,144,362,385,387,263,373,380,61,39,37,0,267,269,291,405,314,17,84,181,78,82,13,312,308,317,14,87]
        self.face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

    def detect(self, frame, visualize=False):
        frame_height, frame_width, channels = frame.shape
        results = self.face_mesh.process(frame)
        multi_face_landmarks = results.multi_face_landmarks

        if multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            landmarks = np.array([(lm.x,lm.y,lm.z) for lm in face_landmarks.landmark])
            landmarks =  landmarks.T

            image_points = landmarks[0:2, self.landmark_points_68].T * np.array([frame_width, frame_height])[None,:]
            return image_points
        else: 
            return None
        
    def plot_markers(self, img, markers, color=(0, 0, 255), radius=3, drawline=False):
        # plot all 68 pts on the face image
        N = markers.shape[0]
        # if N >= 68:
        #     last_point = 68
        for i in range(0, N):
            x = markers[i, 0]
            y = markers[i, 1]
            # cv2.circle(img, (x, y), radius, color)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img, str(i), (x, y), font, 0.3, (255, 0, 0), 1, cv2.LINE_AA)

        if drawline:
            def draw_line(start, end):
                for i in range(start, end):
                    x1 = markers[i, 0]
                    y1 = markers[i, 1]
                    x2 = markers[i + 1, 0]
                    y2 = markers[i + 1, 1]
                    cv2.line(img, (x1, y1), (x2, y2), color)

            draw_line(0, 16)
            draw_line(17, 21)
            draw_line(22, 26)
            draw_line(27, 35)
            draw_line(36, 41)
            draw_line(42, 47)
            draw_line(48, 67)

        return img