#!/usr/bin/env python3

# --------------------------------------------------------
# Copyright (C) 2020 NVIDIA Corporation. All rights reserved.
# NVIDIA Source Code License (1-Way Commercial)
# Code written by Shalini De Mello.
# --------------------------------------------------------

from turtle import numinput
from re import T
import cv2 as cv
import numpy as np
import random
import threading
import pickle
from os import path
import time
import torch
import os
import sys
import copy
import pyrealsense2 as rs
sys.path.append("src")
from process_frame import frame_processor
directions = ['l', 'r', 'u', 'd']
keys = {'u': 82,
        'd': 84,
        'l': 81,
        'r': 83}

global THREAD_RUNNING
global frames, data ,last

def add_kv(list_dict, key, value, num_image_per_point):
    if key in list_dict:
        if not isinstance(value,list):
            list_dict[key].append(value)
        else:
            list_dict[key] += value[-num_image_per_point:]
    else:
        list_dict[key] = list()
        if not isinstance(value,list):
            list_dict[key].append(value)
        else:
            list_dict[key] += value[-num_image_per_point:]

def create_image(mon, direction, i, color, size = 0.5, thickness = 2, target='E', grid=True, total=9, use_last = False):
    global last
    h = mon.h_pixels
    w = mon.w_pixels
    if not use_last: 
        if grid:
            if total == 9:
                col = i % 3
                r = int(i / 3)
                if i != 1:
                    x = int((0.02 + 0.48 * col) * w)
                    y = int((0.02 + 0.48 * r) * h)
                else:
                    x = int((0.02 + 0.48 * col) * w)
                    y = int((0.1) * h)
            elif total == 16:
                col = i % 4
                r = int(i / 4)
                x = int((0.05 + 0.3 * col) * w)
                y = int((0.05 + 0.3 * r) * h)
            elif total == 64:
                col = i % 8
                r = int(i / 8)
                x = int((0.01 + 0.14 * col) * w)
                if r != 0:
                    y = int((0.01 + 0.14 * r) * h)
                else:
                    y = int((0.05 + 0.14 * r) * h)
            elif total == 15:
                col = i % 5
                r = int(i / 5)
                x = int((0.02 + 0.24 * col) * w)
                y = int((0.02 + 0.48 * r) * h)
            else:
                x = int(random.uniform(0.05, 0.95) * w)
                y = int(random.uniform(0.05, 0.95) * h)
        else:
            x = int(random.uniform(0.05, 0.95) * w)
            y = int(random.uniform(0.05, 0.95) * h)
        last = [x,y]
    else:
        x , y = last[0], last[1]

    # compute the ground truth point of regard
    g_t = mon.monitor_to_camera(x, y)
    

    font = cv.FONT_HERSHEY_SIMPLEX
    img = np.ones((h, w, 3), np.float32)
    img[...,0] = 207. / 255.
    img[...,1] = 237. / 255.
    img[...,2] = 199. / 255.


    if direction == 'r' or direction == 'l':
        if direction == 'r':
            cv.putText(img, target, (x, y), font, size, color, thickness, cv.LINE_AA)
        elif direction == 'l':
            cv.putText(img, target, (w - x, y), font, size, color, thickness, cv.LINE_AA)
            img = cv.flip(img, 1)
    elif direction == 'u' or direction == 'd':
        imgT = np.ones((w, h, 3), np.float32)
        imgT[...,0] = 207. / 255.
        imgT[...,1] = 237. / 255.
        imgT[...,2] = 199. / 255.

        if direction == 'd':
            cv.putText(imgT, target, (y, x), font, size, color, thickness, cv.LINE_AA)
        elif direction == 'u':
            cv.putText(imgT, target, (h - y, x), font, size, color, thickness, cv.LINE_AA)
            imgT = cv.flip(imgT, 1)
        img = imgT.transpose((1, 0, 2))

    return img, g_t

def check_len(frames, lens = 10):
    while len(frames) < lens:
        print("len(frames)",len(frames))
        pass

# def grab_img(idx, cap, g_t):
#     global THREAD_RUNNING
#     global frames, data, core
#     while THREAD_RUNNING:
#         _, frame = cap.read()
#         ret_face, normalized_entry, patch_img = core.processors[idx](frame, g_t)
#         if ret_face:
#             frames[idx].append(frame)
#             for key, value in normalized_entry.items():
#                 add_kv(data[idx], key, value)
#             print("For cam%d, the gaze_cam_origin is "%idx,\
#                 normalized_entry["gaze_cam_origin"],flush=True)

class RealSense():
    def __init__(self):
        depth_shape = (1280,720)
        # if path.exists("calib_cam%d_%d.pkl" % (0, depth_shape[0])):
        #      self.realsense_calib = pickle.load(open("calib_cam%d_%d.pkl" % (0, depth_shape[0]), "rb"))
       
        # self.depth_processor = depth_processor()
        self.pipeline = rs.pipeline()
        self.config = rs.config()

        pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        pipeline_profile = self.config.resolve(pipeline_wrapper)
        device = pipeline_profile.get_device()
        device_product_line = str(device.get_info(rs.camera_info.product_line))
        align_to = rs.stream.color
        self.align = rs.align(align_to)
        found_rgb = False
        for s in device.sensors:
            if s.get_info(rs.camera_info.name) == 'RGB Camera':
                found_rgb = True
                break
        if not found_rgb:
            print("The demo requires Depth camera with color sensor")
            exit(0)

        self.config.enable_stream(rs.stream.depth, depth_shape[0], depth_shape[1], rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, depth_shape[0], depth_shape[1], rs.format.bgr8, 30)
        self.pipeline.start(self.config)
        
        
class GrabRealsense(threading.Thread):
    def __init__(self, opt, cam_calibs, real, caps):
        super(GrabRealsense, self).__init__()
        
        self.processors = []
        self.caps = caps
        self.real = real
        for j in range(len(caps)):
            self.processors.append(frame_processor(opt, cam_calibs[j]))
    def run(self):
        global THREAD_RUNNING
        global frames, data, rs_data, g_t
        while THREAD_RUNNING:
            # tic = time.time()
            rs_frames = self.real.pipeline.wait_for_frames()
            # rs_frames = self.real.align.process(rs_frames)
            depth_frame = rs_frames.get_depth_frame()
            color_frame = rs_frames.get_color_frame()
            for j in range(len(self.caps)):
                _, frame =  self.caps[j].read()
                if j == 1:
                    frame = cv.flip(frame, -1)
                frames[j].append(frame)
                # if j == 0:
                #     ret_face, normalized_entry, patch_img = self.processors[j](copy.deepcopy(frame), g_t)
                #     if ret_face:
                #         for key, value in normalized_entry.items():
                #             add_kv(data[j], key, value)

            if not depth_frame or not color_frame:
                continue
            # Convert images to numpy arrays
            depth_image = np.array(depth_frame.get_data())
            color_image = np.array(color_frame.get_data())
            depth_colormap = cv.applyColorMap(cv.convertScaleAbs(depth_image, alpha=0.03), cv.COLORMAP_JET)
            rs_data["depth"].append(depth_image)
            rs_data["color"].append(color_image)
            rs_data["depth_colormap"].append(depth_colormap)
            # cv.imshow("haha", color_image)
            # print(time.time() - tic)

        # self.real.pipeline.stop()    

# class GrabImg(threading.Thread):
#     def __init__(self, processor, idx, cap, mutex):
#         super(GrabImg, self).__init__()
#         self.processor = processor
#         self.idx = idx
#         self.cap = cap
#         self.mutex = mutex
#     def run(self):
#         global THREAD_RUNNING
#         global frames, data, g_t
#         while THREAD_RUNNING:
#             _, frame = self.cap.read()
#             self.mutex.acquire()
#             frames[self.idx].append(frame)
#             self.mutex.release()

#             ret_face, normalized_entry, patch_img = self.processor(copy.deepcopy(frame), g_t)
#             if ret_face:
#                 for key, value in normalized_entry.items():
#                     add_kv(data[self.idx], key, value)
#                 # normalized_entry["gaze_cam_origin"][2,0] -= 10 
#                 # os.system("clear")
#                 # print("For cam%d, the gaze_cam_origin is "%self.idx,\
#                 #     normalized_entry["gaze_cam_origin"].reshape(3),end = "",flush=True)


# def grab_img1(cap, core):
#     global THREAD_RUNNING
#     global frame1s, data
#     while THREAD_RUNNING:
#         _, frame = cap.read()
#         frame1s.append(frame)
#         ret_face, normalized_entry, patch_img = core.processors[1](frame)
#         for key, value in normalized_entry.items():
#             add_kv(data[1], key, value)
#         if ret_face:
#             print("For cam1, the gaze_cam_origin is ",\
#                 normalized_entry["gaze_cam_origin"],flush=True)


def collect_data(subject, caps, mon, opt, cam_calibs, calib_points=9, rand_points=5, view_collect = False):
    global THREAD_RUNNING
    global frames, data, g_t, rs_data
    num_image_per_point = opt.num_image_per_point
    results = []
    data = {}
    frames = {}
    real = RealSense()
    
    # mutex = threading.Lock()

    num_cap = len(caps)
    save_data = {'g_t': [], "depth": [], "color": [], "depth_colormap": []}
    for j in range(num_cap):
        save_data["frame%ds"%j] = []
        data[j] = {}
        frames[j] = []
        results.append({})

    rs_data = {"depth": [], "color": [], "depth_colormap": []}
        # th = GrabImg(opt, cam_calibs[j], j, caps[j], mutex)
        # ths.append(th)
    cv.namedWindow("image", cv.WINDOW_NORMAL)
    cv.setWindowProperty("image", cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)
    for stage, point_num in enumerate([calib_points, rand_points]):
        i = 0
  
        while i < point_num:
            th = GrabRealsense(opt, cam_calibs, real, caps)
                
            # Start the sub-thread, which is responsible for grabbing images
            THREAD_RUNNING = True
            direction = random.choice(directions)
            img, g_t = create_image(mon, direction, i, (0, 0, 0), grid = 1 - stage, total=point_num)
            for j in range(num_cap):
                frames[j] = []
            
                data[j] = {}
            rs_data = {"depth": [], "color": [], "depth_colormap": []}
            th.start()
            cv.imshow('image', img)
            key_press = cv.waitKey(2000)
            if key_press & 0xFF == ord('q'):
                cv.destroyAllWindows()
                THREAD_RUNNING = False
                th.join()
                cv.destroyAllWindows()
                break
            elif key_press > 0 :
                THREAD_RUNNING = False
                th.join()
                continue
            
            img, g_t = create_image(mon, direction, i, (0,  0, 255), grid = 1 - stage, total=point_num, use_last = True)
            cv.imshow('image', img)
            key_press = cv.waitKey(0)
            if key_press == keys[direction]:
                for j in range(num_cap):
                    # while len(frames[j]) < 10:
                    #     time.sleep(0.5)
                    save_data['frame%ds'%j].append(frames[j])
                THREAD_RUNNING = False
                th.join()
                save_data['g_t'].append(g_t)
                save_data['depth'].append(rs_data["depth"])
                save_data['color'].append(rs_data["color"])
                save_data['depth_colormap'].append(rs_data["depth_colormap"])

                img, g_t = create_image(mon, direction, i, (0,  0, 255), thickness = 4,  grid = 1 - stage, total=point_num, use_last = True)
                cv.imshow('image', img)
                cv.waitKey(100)
                time.sleep(0.5)
                i += 1
            elif key_press & 0xFF == ord('q'):
                THREAD_RUNNING = False
                th.join()
                cv.destroyAllWindows()
                break
            else:
                THREAD_RUNNING = False
                th.join()
            # for j in range(num_cap):
            for key, value in data[0].items():
                add_kv(results[0], key, value, num_image_per_point)


    cv.destroyAllWindows()
    img_paths = []
    for j in range(num_cap):
        img_path = 'calibration/%s/%s/cam%d'%(opt.id, subject, j)
        os.makedirs(img_path, exist_ok=True)
        img_paths.append(img_path)
    
        os.makedirs(img_path, exist_ok=True)
    
    target = []
    for j in range(num_cap):
        n = 0 
        for index, frames_ in enumerate(save_data['frame%ds'%j]):
            # print(j, index, len(frames_))
            for k in range(len(frames_) - num_image_per_point, len(frames_)):
                if j == 0:
                    target.append(save_data["g_t"][index])
                frame = frames_[k]
                cv.imwrite(os.path.join(img_paths[j],"%05d.png"%n), frame)
                n += 1

    for key in rs_data.keys():
        n = 0 
        save_path = 'calibration/%s/%s/%s'%(opt.id, subject, key)
        os.makedirs(save_path,exist_ok= True)
        for index, frames_ in enumerate(save_data[key]):
            for k in range(len(frames_) - num_image_per_point, len(frames_)):
                # print(index, len(frames_))
                frame = frames_[k]
                if key == "depth":
                    np.save(os.path.join(save_path,"%05d.npy"%n), frame)
                else:
                    cv.imwrite(os.path.join(save_path,"%05d.png"%n), frame)
                n += 1

    fout = open('calibration/%s/%s/calib_target.pkl' %(opt.id, subject), 'wb')
    pickle.dump(target, fout)
    fout.close()
    return results[0]


def fine_tune(opt, data, core, gaze_network, steps=1000):

    # collect person calibration data
    gaze_network.load_init_networks()
    subject = opt.subject

    if data is None:
        data = core.process_for_finetune(subject)
    
    input_dict_train, input_dict_valid = gaze_network.init_input(data)
    #############
    # Finetuning
    #################
 
    gaze_network.eval()
    valid_loss = gaze_network.test(input_dict_valid).cpu()

    print('%04d> , Validation: %.2f' % (0, valid_loss.item()))

    # gaze_network.optimize_parameters(input_dict_train)

    # valid_loss = gaze_network.test(input_dict_valid).cpu()
    # print('%04d> , Validation: %.2f' % (0, valid_loss.item()))

    for i in range(steps):
        # zero the parameter gradient
        # gaze_network.train()
        gaze_network.train()


        # forward + backward + optimize
        gaze_network.optimize_parameters(input_dict_train)
        if i % int(steps / 10) == int(steps / 10) - 1:
            train_loss = gaze_network.test(input_dict_train).cpu()
            gaze_network.eval()
            valid_loss = gaze_network.test(input_dict_valid).cpu()
            print('%04d> Train: %.2f, Validation: %.2f' %
                  (i+1, train_loss.item(), valid_loss.item()))
    gaze_network.save_networks(subject)
    torch.cuda.empty_cache()

    return gaze_network