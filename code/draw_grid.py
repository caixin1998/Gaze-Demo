import cv2
import numpy as np
import os
def draw_grid(frame_path):
    frame = cv2.imread(frame_path)
    frame_name = frame_path.split('/')[-1]
    pattern_shape = (11, 8)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    pts = np.zeros((pattern_shape[0] * pattern_shape[1], 3), np.float32)
    pts[:, :2] = np.mgrid[0:pattern_shape[0], 0:pattern_shape[1]].T.reshape(-1, 2)

    # cv2.namedWindow("camera", cv2.WINDOW_NORMAL)
    # cv2.setWindowProperty("camera", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # cv2.moveWindow("camera", 1920, 0)
    # capture calibration frames
    obj_points = []  # 3d point in real world space
    img_points = []  # 2d points in image plane.
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    retc, corners = cv2.findChessboardCorners(gray, pattern_shape, None)
    cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    # Draw and display the corners
    cv2.drawChessboardCorners(frame, pattern_shape, corners, retc)
    cv2.imwrite('outputs/%s'%frame_name, frame)

frame_paths = os.listdir('frames')
for frame_path in frame_paths:
    draw_grid(os.path.join('frames', frame_path))