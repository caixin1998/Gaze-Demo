#!/usr/bin/env python3

# --------------------------------------------------------
# Copyright (C) 2020 NVIDIA Corporation. All rights reserved.
# NVIDIA Source Code License (1-Way Commercial)
# Code written by Shalini De Mello.
# --------------------------------------------------------

import cv2 as cv
import numpy as np
import random
import threading
import pickle
import time
import torch
import os
import sys
import copy
sys.path.append("src")
from process_frame import frame_processor
directions = ['l', 'r', 'u', 'd']
keys = {'u': 82,
        'd': 84,
        'l': 81,
        'r': 83}

global THREAD_RUNNING
global frames, data ,last

def add_kv(list_dict, key, value):
    if key in list_dict:
        if not isinstance(value,list):
            list_dict[key].append(value)
        else:
            list_dict[key] += value[-10:]
    else:
        list_dict[key] = list()
        if not isinstance(value,list):
            list_dict[key].append(value)
        else:
            list_dict[key] += value[-10:]

def create_image(mon, direction, i, color, size = 0.5, thickness = 2, target='E', grid=True, total=9, use_last = False):
    global last
    h = mon.h_pixels
    w = mon.w_pixels
    if not use_last: 
        if grid:
            if total == 9:
                row = i % 3
                col = int(i / 3)
                if i != 1:
                    x = int((0.02 + 0.48 * row) * w)
                    y = int((0.02 + 0.48 * col) * h)
                else:
                    x = int((0.02 + 0.48 * row) * w)
                    y = int((0.1) * h)
            elif total == 16:
                row = i % 4
                col = int(i / 4)
                x = int((0.05 + 0.3 * row) * w)
                y = int((0.05 + 0.3 * col) * h)
            elif total == 15:
                row = i % 5
                col = int(i / 5)
                x = int((0.02 + 0.24 * row) * w)
                y = int((0.02 + 0.48 * col) * h)
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

class GrabImg(threading.Thread):
    def __init__(self, processor, idx, cap, mutex):
        super(GrabImg, self).__init__()
        self.processor = processor
        self.idx = idx
        self.cap = cap
        self.mutex = mutex
    def run(self):
        global THREAD_RUNNING
        global frames, data, g_t
        while THREAD_RUNNING:
            _, frame = self.cap.read()
            self.mutex.acquire()
            frames[self.idx].append(frame)
            self.mutex.release()

            ret_face, normalized_entry, patch_img = self.processor(copy.deepcopy(frame), g_t)
            if ret_face:
                for key, value in normalized_entry.items():
                    add_kv(data[self.idx], key, value)
                normalized_entry["gaze_cam_origin"][2,0] -= 10 
                os.system("clear")
                print("For cam%d, the gaze_cam_origin is "%self.idx,\
                    normalized_entry["gaze_cam_origin"].reshape(3),end = "",flush=True)


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
    global frames, data, g_t
    results = []
    data = []
    frames = []
    mutex = threading.Lock()

    num_cap = len(caps)
    calib_data = {'g_t': []}
    ths = []
    processors = []
    for j in range(num_cap):
        calib_data["frame%ds"%j] = []
        data.append({})
        frames.append([])
        results.append({})
        processor = frame_processor(opt, cam_calibs[j])
        processors.append(processor)
        # th = GrabImg(opt, cam_calibs[j], j, caps[j], mutex)
        # ths.append(th)
    
    cv.namedWindow("image", cv.WINDOW_NORMAL)
    cv.setWindowProperty("image", cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)
    for stage, point_num in enumerate([calib_points, rand_points]):
        i = 0
  
        while i < point_num:
            for j in range(num_cap):
                th = GrabImg(processors[j], j, caps[j], mutex)
                ths.append(th)
            # Start the sub-thread, which is responsible for grabbing images
            THREAD_RUNNING = True
            direction = random.choice(directions)
            img, g_t = create_image(mon, direction, i, (0, 0, 0), grid = 1 - stage, total=point_num)
            for j in range(num_cap):
                frames[j] = []
                data[j] = {}
                ths[j].start()
            cv.imshow('image', img)
            key_press = cv.waitKey(2000)
            if key_press & 0xFF == ord('q'):
                cv.destroyAllWindows()
                break
            else:
                time.sleep(1)
            
            img, g_t = create_image(mon, direction, i, (0,  0, 255), grid = 1 - stage, total=point_num, use_last = True)
            cv.imshow('image', img)
            key_press = cv.waitKey(0)
            if key_press == keys[direction]:
                for j in range(num_cap):
                    mutex.acquire()
                    while len(frames[j]) < 10:
                        mutex.release()
                        time.sleep(0.5)
                        mutex.acquire()
                    if mutex.locked():
                        mutex.release()
                    THREAD_RUNNING = False
                    ths[j].join()
                    calib_data['frame%ds'%j].append(frames[j])
                calib_data['g_t'].append(g_t)
                img, g_t = create_image(mon, direction, i, (0,  0, 255), thickness = 4,  grid = 1 - stage, total=point_num, use_last = True)
                cv.imshow('image', img)
                cv.waitKey(100)
                time.sleep(0.5)
                i += 1
            elif key_press & 0xFF == ord('q'):
                THREAD_RUNNING = False
                for j in range(num_cap):
                    ths[j].join()
                cv.destroyAllWindows()
                break
            else:
                THREAD_RUNNING = False
                for j in range(num_cap):
                    ths[j].join()
            for j in range(num_cap):
                for key, value in data[j].items():
                    add_kv(results[j], key, value)

            ths = []


    cv.destroyAllWindows()
    img_paths = []
    for j in range(num_cap):
        img_path = 'calibration/%s_calib/cam%d'%(subject, j)
        os.makedirs(img_path, exist_ok=True)
        img_paths.append(img_path)
    target = []
    for j in range(num_cap):
        n = 0 
        for index, frames_ in enumerate(calib_data['frame%ds'%j]):
            print(index, len(frames_))
            for k in range(len(frames_) - 10, len(frames_)):
                frame = frames_[k]
                g_t = calib_data['g_t'][index]
                target.append(g_t)
            # print(os.path.join(imgs_path,"%05d.png"%n))
                cv.imwrite(os.path.join(img_paths[j],"%05d.png"%n), frame)
                n += 1

    # print(target)
    fout = open('calibration/%s_calib_target.pkl' % subject, 'wb')
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

    for i in range(steps):
        # zero the parameter gradient
        gaze_network.train()

        # forward + backward + optimize
        gaze_network.optimize_parameters(input_dict_train)
        if i % 100 == 99:
            train_loss = gaze_network.test(input_dict_train).cpu()
            gaze_network.eval()
            valid_loss = gaze_network.test(input_dict_valid).cpu()
            print('%04d> Train: %.2f, Validation: %.2f' %
                  (i+1, train_loss.item(), valid_loss.item()))
    gaze_network.save_networks(subject)
    torch.cuda.empty_cache()

    return gaze_network