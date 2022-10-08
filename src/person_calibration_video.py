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
import multiprocessing as mp
import os
import sys
import copy
import pyrealsense2 as rs
import pdb
from queue import Queue
import skvideo.io
import h5py
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

            elif total == 30:
                col = i % 6
                r = int(i / 6)
                x = int((0.025 + 0.19 * col) * w)
                y = int((0.02 + 0.24 * r) * h)
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
    # g_t = mon.monitor_to_camera(x, y)

    g_t = np.array([x,y])

    font = cv.FONT_HERSHEY_SIMPLEX
    img = np.ones((h, w, 3), np.float32)
    img[...,0] = 207. / 255.
    img[...,1] = 237. / 255.
    img[...,2] = 199. / 255.


    if direction == 'r' or direction == 'l':
        if direction == 'r':
            # cv.circle(img, (x,y),10, (255,0,0),-1)
            cv.putText(img, target, (x-5, y + 5), font, size, color, thickness, cv.LINE_AA)
        elif direction == 'l':
            # cv.circle(img, (w - x, y),10, (255,0,0),-1)
            cv.putText(img, target, (w - x - 5, y + 5), font, size, color, thickness, cv.LINE_AA)
            img = cv.flip(img, 1)
    elif direction == 'u' or direction == 'd':
        imgT = np.ones((w, h, 3), np.float32)
        imgT[...,0] = 207. / 255.
        imgT[...,1] = 237. / 255.
        imgT[...,2] = 199. / 255.

        if direction == 'd':
            # cv.circle(imgT, (y,x),10, (255,0,0),-1)
            cv.putText(imgT, target, (y - 5, x + 5), font, size, color, thickness, cv.LINE_AA)
        elif direction == 'u':
            # cv.circle(imgT, (h-y,x),10, (255,0,0),-1)
            cv.putText(imgT, target, (h - y - 5, x + 5), font, size, color, thickness, cv.LINE_AA)
            imgT = cv.flip(imgT, 1)
        img = imgT.transpose((1, 0, 2))

    return img, g_t


class GrabImg(threading.Thread):
    def __init__(self, opt, cam_calibs, queues, rs_data, num_image_per_point):
        super(GrabImg, self).__init__()
        self.processors = []
        for j in range(len(opt.cam_idx)):
            self.processors.append(frame_processor(opt, cam_calibs[j]))
        self.queues = queues
        self.rs_data = rs_data
        self.num_image_per_point = num_image_per_point
    def run(self):
        global THREAD_RUNNING
        global  data, g_t, idx
        rs_data = self.rs_data

        while THREAD_RUNNING:

            for j in range(len(self.processors)):
                frame =  self.queues[j].get()
                # if j == 1:
                #     frame["frame"] = cv.flip(frame["frame"], -1)
                
                rs_data["frame%ds"%j].append(frame["frame"])
                rs_data["time%ds"%j].append(frame["time"])
                # if j == 0:
                #     ret_face, normalized_entry, patch_img = self.processors[j](copy.deepcopy(frame), g_t)
                #     if ret_face:
                #         for key, value in normalized_entry.items():
                #             add_kv(data[j], key, value, 1)
                #         print("For cam%d, the gaze_cam_origin is "%0,\
                # normalized_entry["gaze_cam_origin"].reshape(3),end = "\n",flush=True)
            rs_data["depth"].append(self.queues[-3].get())
            rs_data["color"].append(self.queues[-2].get())
            # rs_data["depth_colormap"].append(self.queues[-2].get())
            rs_data["time"].append(self.queues[-1].get())

            for key in rs_data.keys():
                rs_data[key] = rs_data[key][-self.num_image_per_point:]
        # print(rs_data)

def write_video_process(cap_id, collect, queue, video_path):
    # writer = skvideo.io.FFmpegWriter(os.path.join(video_path, "cam%d.mp4"%cap_id), outputdict={
    #     '-r': "30",
    #     '-vcodec': 'libx264',  #use the h.264 codec
    #     '-crf': '0',           #set the constant rate factor to 0, which is lossless
    #     '-preset':'slow'   #the slower the better compression, in princple, try 
    #                      #other options see https://trac.ffmpeg.org/wiki/Encode/H.264
    #     }) 
    # writer = skvideo.io.FFmpegWriter(os.path.join(video_path, "cam%d.mp4"%cap_id), outputdict={
    #     '-r': "30",
    #     "-hwaccel": "cuda",
    #     "-hwaccel_output_format": "cuda",
    #     "-vcodec": "h264_nvenc",
    #     "-b:v": "10M",
    #     }) 
    import ffmpeg
    save_path = os.path.join(video_path, "cam%d.mp4"%cap_id)
    print("cap_id:", cap_id)

    writer = (
        ffmpeg
        .input('pipe:', v=0, hide_banner=None, nostats=None, vsync="passthrough", r=30, format='rawvideo', pix_fmt='bgr24', s='{}x{}'.format(1920, 1080))
        .output(save_path, vcodec='h264_nvenc', preset='p6', profile='main', crf=0)#, video_bitrate='10M')
        # .output(save_path, vcodec='libx264')

        .overwrite_output()
        .run_async(pipe_stdin=True)
    )

    print("cap_id:", cap_id)
    timestamps = list() 
    while collect.value:
        frame_dict =  queue.get()
        # print(queue.qsize())
        frame = frame_dict["frame"]
        # if cap_id == 1:
        #     frame = cv.flip(frame, -1)
        # frame = frame[...,::-1]
        # writer.writeFrame(frame)
        writer.stdin.write(frame.tobytes())
        timestamps += [frame_dict["time"]]
    timestamps = np.array(timestamps)
    np.savetxt(os.path.join(video_path, "time%d.txt"%cap_id), timestamps, fmt = "%.3f")
    # writer.close()
    
    writer.stdin.close()
    writer.wait()


def write_videos_process(cap_ids, collect, queues, video_path):
    import ffmpeg
    writers = []
    timestamps = []
    for cap_id in cap_ids:
        save_path = os.path.join(video_path, "cam%d.mp4"%cap_id)
        print("cap_id:", cap_id)
        writers += [(
            ffmpeg
            .input('pipe:', v=0, hide_banner=None, nostats=None, vsync="passthrough", r=30, format='rawvideo', pix_fmt='bgr24', s='{}x{}'.format(1920, 1080))
           .output(save_path, vcodec='h264_nvenc', preset='hq', profile='main', crf=0, video_bitrate='9M')
            # .output(save_path, vcodec='libx264')

            .overwrite_output()
            .run_async(pipe_stdin=True)
        )]
        print("cap_id:", cap_id)
        timestamps += [list()] 
    while collect.value:
        for i, queue in enumerate(queues):
            frame_dict =  queue.get()
        # print(queue.qsize())
            frame = frame_dict["frame"]
            # if i == 1:
            #     frame = cv.flip(frame, -1)
        # frame = frame[...,::-1]
        # writer.writeFrame(frame)
            writers[i].stdin.write(frame.tobytes())
            timestamps[i] += [frame_dict["time"]]

    for i, timestamp in enumerate(timestamps):
        timestamp = np.array(timestamp)
        np.savetxt(os.path.join(video_path, "time%d.txt"%cap_id), timestamp, fmt = "%.3f")
        writers[i].stdin.close()
        writers[i].wait()

def write_color_process(queue, collect, color_path):
    save_path = os.path.join(color_path, "color.mp4")
    writer = cv.VideoWriter(save_path, cv.VideoWriter_fourcc('m','p','4','v'), 30, (1280,720), True)
    # writer = skvideo.io.FFmpegWriter(save_path, outputdict={
    #     '-r': "30",
    #     '-vcodec': 'libx264',  #use the h.264 codec
    #     '-crf': '0',           #set the constant rate factor to 0, which is lossless
    #     '-preset':'fast'   #the slower the better compression, in princple, try 
    #                      #other options see https://trac.ffmpeg.org/wiki/Encode/H.264
    #     }) 
    while collect.value:
        frame =  queue.get()
        # print('color:', queue.qsize())
        writer.write(frame)
    writer.release()



def write_depth_process(queues, collect, depth_path):
    os.makedirs(os.path.join(depth_path, 'depth'), exist_ok=True)
    import ffmpeg
    # save_path = os.path.join(depth_path, "color.mp4")
    # writer = (
    #     ffmpeg
    #     .input('pipe:', v=0, hide_banner=None, nostats=None, vsync="passthrough", r=30, format='rawvideo', pix_fmt='bgr24', s='{}x{}'.format(1280, 720))
    #     .output(save_path, vcodec='h264_nvenc', preset='hq', profile='main', video_bitrate='10M')
    #     .overwrite_output()
    #     .run_async(pipe_stdin=True)
    # )
    timestamps = list() 
    i = 0
    while collect.value:
        # frame =  queues[-3].get()
        # writer.stdin.write(frame.tobytes())

        depth = queues[-2].get()
        with h5py.File(os.path.join(depth_path,"depth","%05d.h5"%i), "w") as f:
            f.create_dataset("depth", data = depth, compression = 'lzf')
        i += 1
        # print('depth:', queues[-2].qsize())
        timestamps += [queues[-1].get()]
    # writer.stdin.close()
    # writer.wait()
    timestamps = np.array(timestamps)
    np.savetxt(os.path.join(depth_path, "time_realsense.txt"), timestamps, fmt = "%.3f")


def collect_data(subject, queues, mon, opt, cam_calibs, calib_points=9, rand_points=5, view_collect = False):
    global THREAD_RUNNING
    global  data, g_t ,idx
    num_image_per_point = opt.num_image_per_point
    data = {}
    num_cap = len(opt.cam_idx)
    collect =  mp.Value("i", 1)
    processes = []   
    # img_path = '/home/caixin/nas_data/VIPLIIGaze/calibration/%s/%s'%(opt.id, subject)
    img_path = 'calibration/%s/%s'%(opt.id, subject)

    os.makedirs(img_path, exist_ok=True)
    for j in range(num_cap):
        processes.append(mp.Process(target = write_video_process, args = (j,collect,queues[j],img_path,)))
    processes.append(mp.Process(target = write_depth_process, args = (queues[-2:], collect, img_path,)))
    # processes.append(mp.Process(target = write_videos_process, args = (list(range(num_cap)),collect,queues[:num_cap],img_path,)))
    processes.append(mp.Process(target = write_color_process, args = (queues[-3], collect, img_path,)))

    time_data = {}
    time_data['g_t'] = []
    time_data['time'] = []

        # th = GrabImg(opt, cam_calibs[j], j, caps[j], mutex)
        # ths.append(th)
    # cv.setWindowProperty("image", cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)
    idx = 0

    for process in processes:
        process.daemon = True
        process.start()
    mark = 0
    cv.namedWindow("image", cv.WINDOW_NORMAL)
    cv.setWindowProperty("image", cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)
    for stage, point_num in enumerate([calib_points, rand_points]):
        i = 0
        
        while i < point_num:
            # Start the sub-thread, which is responsible for grabbing images
            THREAD_RUNNING = True
            direction = random.choice(directions)
            img, g_t = create_image(mon, direction, i, (0, 0, 0), grid = 1 - stage, total=point_num)
            for j in range(num_cap):
                data[j] = {}
            cv.imshow('image', img)
            if stage == 0:
                key_press = cv.waitKey(1500)
            else:
                key_press = cv.waitKey(1900)

            if key_press & 0xFF == ord('q'):
                mark = 1
                cv.destroyAllWindows()
                break
            elif key_press > 0 :
                continue
            img, g_t = create_image(mon, direction, i, (0,  0, 255), grid = 1 - stage, total=point_num, use_last = True)
            cv.imshow('image', img)
            key_press = cv.waitKey(0)
            if key_press == keys[direction]:

                time_data['g_t'].append(g_t)
                time_data['time'].append(time.time())
                img, g_t = create_image(mon, direction, i, (0,  0, 255), thickness = 4,  grid = 1 - stage, total=point_num, use_last = True)
                cv.imshow('image', img)
                cv.waitKey(50)

                time.sleep(0.3)
                i += 1
                idx += 1
            elif key_press & 0xFF == ord('q'):
                mark = 1
                cv.destroyAllWindows()
                break
            # for j in range(num_cap):
        
        if mark:
            break
    collect.value = 0
    cv.destroyAllWindows()

    for process in processes:
        process.join()

    target = np.array(time_data['g_t'])
    times = np.array(time_data['time'])

    np.savetxt('calibration/%s/%s/calib_target.txt' %(opt.id, subject), target) 
    np.savetxt('calibration/%s/%s/times.txt' %(opt.id, subject), times, fmt="%.3f")



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