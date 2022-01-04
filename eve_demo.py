

import time
import cv2 as cv
import numpy as np
from os import path
from subprocess import call
import pickle
import sys
import torch
import os
import warnings
import cv2 as cv
from options.base_options import BaseOptions
warnings.filterwarnings("ignore")
sys.path.append("src")
from monitor import monitor
from camera import cam_calibrate
from person_calibration import collect_data, fine_tune
from core import process_core
from models import create_model
#################################
# Start camera
#################################
#cal = 1

opt = BaseOptions().parse() 

import h5py
cam_idx = opt.cam_idx
cam_cap = cv.VideoCapture('eve_test/webcam_c.mp4')
cam_calib = {'mtx': np.eye(3), 'dist': np.zeros((1, 5))}

with h5py.File("eve_test/webcam_c.h5", "r") as f:
    cam_calib['mtx'] = f['camera_matrix'][:]
    cam_calib['face_g'] = f['face_g_tobii']['data'][:]

# collect person calibration data and fine-
# tune gaze network

model = create_model(opt)
mon = monitor()
processor = process_core(opt, cam_calib)

model.load_networks()

#################################
# Run on live webcam feed and
# show point of regard on screen
#################################
processor.process(cam_cap, model)
