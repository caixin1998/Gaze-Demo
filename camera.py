#!/usr/bin/env python3

# --------------------------------------------------------
# Copyright (C) 2020 NVIDIA Corporation. All rights reserved.
# NVIDIA Source Code License (1-Way Commercial)
# Code written by Shalini De Mello.
# --------------------------------------------------------

from typing import Pattern
import cv2
import numpy as np
import pickle
from subprocess import call
def cam_calibrate(cam_idx, cap, cam_calib):

    # termination criteria
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
    frames = []
    while True:
        ret, frame = cap.read()
        frame_copy = frame.copy()

        corners = []
        if ret:
            # cv2.imshow('camera', frame)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            retc, corners = cv2.findChessboardCorners(gray, pattern_shape, None)
            if cv2.waitKey(1)&0xFF==ord('q'):
                print("Calibrating camera...")
                cv2.destroyAllWindows()
                break
            print(retc)
            if retc:
                print('collect!')
                cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                # Draw and display the corners
                cv2.drawChessboardCorners(frame_copy, pattern_shape, corners, ret)

                cv2.imshow('points', frame_copy)
                key_press = cv2.waitKey(0)
                # s to save, c to continue, q to quit
                if  key_press & 0xFF == ord('s'):
                    img_points.append(corners)
                    obj_points.append(pts)
                    frames.append(frame)
                    cv2.destroyWindow('points')
                elif key_press & 0xFF == ord('c'):
                    cv2.destroyWindow('points')
                    continue
                elif key_press & 0xFF == ord('q'):
                    print("Calibrating camera...")
                    cv2.destroyAllWindows()
                    break

    # compute calibration matrices

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, frames[0].shape[0:2], None, None)

    # check
    error = 0.0
    for i in range(len(frames)):
        proj_imgpoints, _ = cv2.projectPoints(obj_points[i], rvecs[i], tvecs[i], mtx, dist)
        error += (cv2.norm(img_points[i], proj_imgpoints, cv2.NORM_L2) / len(proj_imgpoints))
    print("Camera calibrated successfully, total re-projection error: %f" % (error / len(frames)))

    cam_calib['mtx'] = mtx
    cam_calib['dist'] = dist
    print("Camera parameters:")
    print(cam_calib)

    pickle.dump(cam_calib, open("calib_cam%d_1920.pkl" % (cam_idx), "wb"))

if __name__ == '__main__':
    cam_idx = 0
    call('v4l2-ctl -d /dev/video%d -c brightness=128' % cam_idx, shell=True)
    call('v4l2-ctl -d /dev/video%d -c contrast=128' % cam_idx, shell=True)
    call('v4l2-ctl -d /dev/video%d -c sharpness=128' % cam_idx, shell=True)
    cap = cv2.VideoCapture(cam_idx)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cam_calib = {'mtx': np.eye(3), 'dist': np.zeros((1, 5))}
    cam_calibrate(cam_idx, cap, cam_calib)