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
from options.base_options import BaseOptions
from options.finetune_options import FinetuneOptions

warnings.filterwarnings("ignore")
sys.path.append("src")
from monitor import monitor
from camera import cam_calibrate
from person_calibration_video import collect_data, fine_tune
from core import process_core
from models import create_model
import multiprocessing as mp
import pyrealsense2 as rs
import time
import pdb
#################################
# Start camera
#################################
#cal = 1
def image_put(cam_idx, camera_size, q):
        cap = cv.VideoCapture(cam_idx)
        cap.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        cap.set(cv.CAP_PROP_FRAME_WIDTH, camera_size[0])
        cap.set(cv.CAP_PROP_FRAME_HEIGHT, camera_size[1])
        
        while cap.isOpened():
            # print('cap.read()[0]:', cap.read()[0])
            ret, frame = cap.read()
            if cam_idx == 4:
                frame = cv.flip(frame, -1)
            # print('ret:', ret)
            if ret:
                q.put(frame)
                q.get() if q.qsize() > 1 else time.sleep(0.01)

def depth_put(queue, depth_shape = (1920,1080)):
        pipeline = rs.pipeline()
        config = rs.config()

        pipeline_wrapper = rs.pipeline_wrapper(pipeline)
        pipeline_profile = config.resolve(pipeline_wrapper)
        device = pipeline_profile.get_device()
        device_product_line = str(device.get_info(rs.camera_info.product_line))
        align_to = rs.stream.color
        align = rs.align(align_to)
        found_rgb = False
        for s in device.sensors:
            if s.get_info(rs.camera_info.name) == 'RGB Camera':
                found_rgb = True
                break
        if not found_rgb:
            print("The demo requires Depth camera with color sensor")
            exit(0)
    
        config.enable_stream(rs.stream.color, depth_shape[0], depth_shape[1], rs.format.bgr8, 30)
        profile = pipeline.start(config)

        depth_sensor = profile.get_device().first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()
        clipping_distance_in_meters = 1.5 #1 meter
        clipping_distance = clipping_distance_in_meters / depth_scale

        while True:
            rs_frames = pipeline.wait_for_frames()
            color_frame = rs_frames.get_color_frame()
            if not color_frame:
                continue
            # Convert images to numpy arrays
            color_image = np.array(color_frame.get_data())
            # depth_image_clip = np.where((depth_image > clipping_distance),clipping_distance, depth_image)
            # depth_image_3d = np.dstack((depth_image,depth_image,depth_image)) #depth image is 1 channel, color is 3 channels
            # bg_removed = np.where((depth_image_3d > clipping_distance) | (depth_image_3d <= 0), grey_color, color_image)
            # depth_colormap = cv.applyColorMap(cv.convertScaleAbs(depth_image_clip, alpha=0.03), cv.COLORMAP_JET)
            queue.put(color_image)
            # queues[2].put(depth_colormap)

            # for q in queues:
            queue.get() if queue.qsize() > 1 else time.sleep(0.01)

def show(queues):
    cv.namedWindow('RealSense', cv.WINDOW_AUTOSIZE)
    while True:
        frames = []
        for queue in queues:
            frames.append(cv.resize(queue.get(),(960,540)))
        images = np.vstack((np.hstack((frames[0],frames[1])), np.hstack((frames[2],frames[3]))))
        
        cv.imshow('RealSense', images)

        key = cv.waitKey(1)
    # cv2.imwrite(os.path.join("frames","%05d.png"%i), frame)
        if key&0xFF==ord('q'):
            break

if __name__ == "__main__":
    mp.set_start_method(method='spawn')
    queues = [mp.Queue(maxsize=2) for _ in range(4)]
    processes = []
    for i in range(3):
        processes.append(mp.Process(target=image_put, args=(i + 3, (1920,1080), queues[i])))
    processes.append(mp.Process(target=depth_put, args=(queues[-1],)))

    for process in processes:
        process.daemon = True
        process.start()

    show(queues)