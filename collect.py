

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
                q.put({"frame":frame, "time": time.time()})
                # print("cam_idx:", cam_idx, q.qsize())

                if q.qsize() > 1:
                    # print("!!!!!!!!!")
                    pass
                else:
                    time.sleep(0.01)

                # q.get() if q.qsize() > 1 else time.sleep(0.01)
                # print("cam_idx:", cam_idx, q.size())

def depth_put(queues, depth_shape = (1280,720)):
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
    
        config.enable_stream(rs.stream.depth, depth_shape[0], depth_shape[1], rs.format.z16, 30)
        config.enable_stream(rs.stream.color, depth_shape[0], depth_shape[1], rs.format.bgr8, 30)
        profile = pipeline.start(config)

        depth_sensor = profile.get_device().first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()
        clipping_distance_in_meters = 1.5 #1 meter
        clipping_distance = clipping_distance_in_meters / depth_scale

        while True:
            tic = time.time()
            rs_frames = pipeline.wait_for_frames()
            rs_frames = align.process(rs_frames)
            depth_frame = rs_frames.get_depth_frame()
            color_frame = rs_frames.get_color_frame()

            if not depth_frame or not color_frame:
                continue
            # Convert images to numpy arrays
            depth_image = np.array(depth_frame.get_data())
            color_image = np.array(color_frame.get_data())
            
            # depth_image_clip = np.where((depth_image > clipping_distance),clipping_distance, depth_image)
            # depth_image_3d = np.dstack((depth_image,depth_image,depth_image)) #depth image is 1 channel, color is 3 channels
            # bg_removed = np.where((depth_image_3d > clipping_distance) | (depth_image_3d <= 0), grey_color, color_image)
            # depth_colormap = cv.applyColorMap(cv.convertScaleAbs(depth_image_clip, alpha=0.03), cv.COLORMAP_JET)
            queues[1].put(depth_image)
            queues[0].put(color_image)
            # queues[2].put(depth_colormap)
            queues[2].put(time.time())
            # print("color_image:", queues[0].qsize())
            
            for i, q in enumerate(queues):
                if q.qsize() > 1:
                    # print("!!!!!!!!!", i)
                    pass
                else:
                    time.sleep(0.01)
            # print(time.time() - tic)
def draw_tips(img_path):
    # text = "请调整您的座椅至（x=%d, y=%d, z=%d）处"
    font = cv.FONT_HERSHEY_SIMPLEX
    img = cv.imread(img_path)
    img = cv.resize(img, (1920, 1080))
    # img = np.ones((1080, 1920, 3), np.float32)
    # img[...,0] = 207. / 255.
    # img[...,1] = 237. / 255.
    # img[...,2] = 199. / 255.
    # cv.putText(img, text, (300,500), font, 80, (0,  0, 255), 4, cv.LINE_AA)
    # cv.
    cv.namedWindow("image1", cv.WINDOW_NORMAL)
    cv.setWindowProperty("image1", cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)
    cv.imshow('image1', img)
    if cv.waitKey(0):
        cv.destroyAllWindows()

def show(queues):
    cv.namedWindow('RealSense', cv.WINDOW_NORMAL)
    cv.setWindowProperty("RealSense", cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)
    
    while True:
        frames = []
        for queue in queues[:-3]:
            frames.append(cv.resize(queue.get()["frame"],(960,540)))
        frames.append(cv.resize(queues[-3].get(),(960,540)))
        queues[-2].get()
        queues[-1].get()
        images = np.vstack((np.hstack((frames[0],frames[1])), np.hstack((frames[2],frames[3]))))
        
        cv.imshow('RealSense', images)

        key = cv.waitKey(1)
    # cv2.imwrite(os.path.join("frames","%05d.png"%i), frame)
        if key&0xFF==ord('q'):
            break

if __name__ == '__main__':
    opt = FinetuneOptions().parse() 
    torch.backends.cudnn.benchmark = True
    cam_indices, cam_caps, cam_calibs = [], [], []
    for i,cam_idx in enumerate(opt.cam_idx):
        cam_indices += [cam_idx]
        cam_caps += [cv.VideoCapture(cam_idx)]
        cam_caps[i].set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        cam_caps[i].set(cv.CAP_PROP_FRAME_WIDTH, opt.camera_size[0])
        cam_caps[i].set(cv.CAP_PROP_FRAME_HEIGHT, opt.camera_size[1])
    # calibrate camera
        cam_calib = {'mtx': np.eye(3), 'dist': np.zeros((1, 5))}

        if path.exists("intrinsic/%s/calib_cam%d.pkl" % (opt.id, cam_idx)):
            cam_calib = pickle.load(open("intrinsic/%s/calib_cam%d.pkl" % (opt.id, cam_idx), "rb"))
            print('cam_calib:', cam_calib)
        else:
            print("Calibrate camera once. Print pattern.png, paste on a clipboard, show to camera and capture non-blurry images in which points are detected well.")
            print("Press s to save frame, c to continue, q to quit")
            cam_calibrate(cam_idx, cam_caps[i], cam_calib)
        cam_calibs += [cam_calib]

    for cap in cam_caps:
        cap.release()

    mp.set_start_method(method='spawn')
    queues = [mp.Queue(maxsize=2) for _ in opt.cam_idx]
    processes = []
    for queue, cam_id in zip(queues, opt.cam_idx):
        processes.append(mp.Process(target=image_put, args=(cam_id, opt.camera_size, queue)))
    if opt.depth:
        queues += [mp.Queue(maxsize=2) for _ in range(3)]
        processes.append(mp.Process(target=depth_put, args=(queues[-3:],)))
    for process in processes:
        process.daemon = True
        process.start()
    # collect person calibration data and fine-
    # tune gaze network
    subject = input('Enter subject name: ')
    img_list = [os.path.join("tips", img) for img in os.listdir("tips")]
    img_list.sort()
    # print(img_list)
    subject += "+%d"%opt.k
    opt.subject = subject
    i = 0
    for z in range(3):
        # for y in range(2):
        for (x,y) in [(0,0),(1,0),(2,0),(2,1),(1,1),(0,1)]:
            draw_tips(img_list[i])
            show(queues)
            subject_i = '%s_%d'%(subject, i)
            os.makedirs('calibration/%s/%s'%(opt.id, subject_i), exist_ok= True)
        # position = input('Enter your position (x y z) :')
            position = "%d %d %d"%(x,y,z)
            with open('calibration/%s/%s/position.txt' %(opt.id, subject_i), 'w') as f:
                f.write(position) 


            # calib_list  = [cal_sample.split('_')[0] for cal_sample in os.listdir("calibration") ]
            # data = None
            # model = create_model(opt)
            mon = monitor("/home/caixin/tnm-opencv/data/%s/cam%s/opt.txt"%(opt.id, opt.cam_idx[0]))
            # core = process_core(opt, cam_calibs)
            # if opt.do_collect:
            
            collect_data(subject_i, queues, mon, opt, cam_calibs, calib_points=opt.k, rand_points=16, view_collect = False)
            i += 1
    # if opt.do_finetune or opt.do_collect:
    #     model = fine_tune(opt, data, core, model, steps = opt.step)
    # else:
    #     model.load_networks()


    #################################
    # Run on live webcam feed and
    # show point of regard on screen
    #################################
    

    # core.process(queues[:1], model)
