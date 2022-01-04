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

directions = ['l', 'r', 'u', 'd']
keys = {'u': 82,
        'd': 84,
        'l': 81,
        'r': 83}

global THREAD_RUNNING
global frames

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


def grab_img(cap):
    global THREAD_RUNNING
    global frames
    while THREAD_RUNNING:
        _, frame = cap.read()
        frames.append(frame)


def collect_data(subject, cap, mon, calib_points=9, rand_points=5):
    global THREAD_RUNNING
    global frames

    cv.namedWindow("image", cv.WINDOW_NORMAL)
    cv.setWindowProperty("image", cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)

    calib_data = {'frames': [], 'g_t': []}

    i = 0
    while i < calib_points:

        # Start the sub-thread, which is responsible for grabbing images
        frames = []
        THREAD_RUNNING = True
        th = threading.Thread(target=grab_img, args=(cap,))
        th.start()
        direction = random.choice(directions)
        img, g_t = create_image(mon, direction, i, (0, 0, 0), grid=True, total=calib_points)
        cv.imshow('image', img)
        cv.waitKey(1000)
        time.sleep(1)
        img, g_t = create_image(mon, direction, i, (0,  0, 255), grid=True, total=calib_points, use_last = True)
        cv.imshow('image', img)
        key_press = cv.waitKey(0)
        if key_press == keys[direction]:
            while len(frames) < 10:
                pass
            THREAD_RUNNING = False
            th.join()
            calib_data['frames'].append(frames)
            calib_data['g_t'].append(g_t)
            img, g_t = create_image(mon, direction, i, (0,  0, 255), thickness = 4,  grid=True, total=calib_points, use_last = True)
            cv.imshow('image', img)
            cv.waitKey(10)
            time.sleep(0.5)
            i += 1
        elif key_press & 0xFF == ord('q'):
            cv.destroyAllWindows()
            break
        else:
            THREAD_RUNNING = False
            th.join()

        # time.sleep(0.5)

    i = 0
    while i < rand_points:

        # Start the sub-thread, which is responsible for grabbing images
        frames = []
        THREAD_RUNNING = True
        th = threading.Thread(target=grab_img, args=(cap,))
        th.start()
        direction = random.choice(directions)
        img, g_t = create_image(mon, direction, i, (0, 0, 0), grid=False, total=rand_points)
        cv.imshow('image', img)
        cv.waitKey(1000)
        time.sleep(1)
        img, g_t = create_image(mon, direction, i, (0,  0, 255), grid=False, total=rand_points, use_last = True)
        cv.imshow('image', img)
        key_press = cv.waitKey(0)
        if key_press == keys[direction]:
            while len(frames) < 10:
                pass
            THREAD_RUNNING = False
            th.join()
            calib_data['frames'].append(frames)
            calib_data['g_t'].append(g_t)
            img, g_t = create_image(mon, direction, i, (0,  0, 255), grid=False, thickness = 4,  total=rand_points,  use_last = True)
            cv.imshow('image', img)
            cv.waitKey(10)
            time.sleep(0.5)
            i += 1
        elif key_press & 0xFF == ord('q'):
            cv.destroyAllWindows()
            break
        else:
            THREAD_RUNNING = False
            th.join()
    cv.destroyAllWindows()
    
    imgs_path = 'calibration/%s_calib'% subject
    os.makedirs(imgs_path, exist_ok=True)
    target = []
    n = 0

    for index, frames in enumerate(calib_data['frames']):
        for i in range(len(frames) - 10, len(frames)):
            frame = frames[i]
            g_t = calib_data['g_t'][index]
            target.append(g_t)
            # print(os.path.join(imgs_path,"%05d.png"%n))
            cv.imwrite(os.path.join(imgs_path,"%05d.png"%n), frame)

            n += 1
    # print(target)
    fout = open('calibration/%s_calib_target.pkl' % subject, 'wb')
    pickle.dump(target, fout)
    fout.close()
    return calib_data


def fine_tune(opt, core, gaze_network, steps=1000):

    # collect person calibration data
    gaze_network.load_init_networks()
    subject = opt.subject
    pre_data = core.process_for_finetune(subject)
    
    input_dict_train, input_dict_valid = gaze_network.init_input(pre_data)
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