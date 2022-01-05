

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

cam_indices, cam_caps, cam_calibs = [], [], []
for i,cam_idx in enumerate(opt.cam_idx):
    cam_indices += [cam_idx]
    cam_caps += [cv.VideoCapture(cam_idx)]
    cam_caps[i].set(cv.CAP_PROP_FRAME_WIDTH, opt.camera_size[0])
    cam_caps[i].set(cv.CAP_PROP_FRAME_HEIGHT, opt.camera_size[1])

# calibrate camera
    cam_calib = {'mtx': np.eye(3), 'dist': np.zeros((1, 5))}

    if path.exists("calib_cam%d_%d.pkl" % (cam_idx, opt.camera_size[0])):
        cam_calib = pickle.load(open("calib_cam%d_%d.pkl" % (cam_idx, opt.camera_size[0]), "rb"))
        print('cam_calib:', cam_calib)
    else:
        print("Calibrate camera once. Print pattern.png, paste on a clipboard, show to camera and capture non-blurry images in which points are detected well.")
        print("Press s to save frame, c to continue, q to quit")
        cam_calibrate(cam_idx, cam_caps[i], cam_calib)
    
    cam_calibs += [cam_calib]



# collect person calibration data and fine-
# tune gaze network
subject = input('Enter subject name: ')
subject += "+%d"%opt.k
opt.subject = subject
calib_list  = [cal_sample.split('_')[0] for cal_sample in os.listdir("calibration") ]
data = None
model = create_model(opt)
mon = monitor()
processor = process_core(opt, cam_calibs)

if opt.do_collect:
    data = collect_data(subject, cam_caps, processor, mon, calib_points=1, rand_points=1, view_collect = False)

if opt.do_finetune or opt.do_collect:
    model = fine_tune(opt, data, processor, model, steps=1000)
else:
    model.load_init_networks()


#################################
# Run on live webcam feed and
# show point of regard on screen
#################################
processor.process(cam_caps, model)
