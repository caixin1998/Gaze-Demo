
import cv2 as cv
from subprocess import call
import numpy as np
from os import path
import pickle
import sys
import os
import torch

sys.path.append("src")
from undistorter import Undistorter
from KalmanFilter1D import Kalman1D
import multiprocessing as mp
# from face import Face
from landmarks import landmarks
from head import PnPHeadPoseEstimator
from normalization import normalize,vector_to_pitchyaw
#plane_z = -20.0
#plane_y = -15.0#15.0
from monitor import monitor
from process_frame import frame_processor
from utils import  draw_gaze, to_screen_coordinates
# from tensor_utils import to_screen_coordinates 
import time
import cv2 as cv
import random

def add_kv(list_dict, key, value):
    if key in list_dict:
        list_dict[key].append(value)
    else:
        list_dict[key] = list()
        list_dict[key].append(value)




class process_core:

    def __init__(self, opt, cam_calibs):
        
        self.mon = monitor("/home/caixin/tnm-opencv/data/%s/cam%s/opt.txt"%(opt.id, opt.cam_idx[0]))
        self.cell_x = 3
        self.cell_y = 3
        self.start  = 0
        self.mark = -1
        self.opt = opt
        self.point_list = []
        x_list = [(2 * i + 1) * self.mon.w_pixels / (2 * self.cell_x) for i in range(self.cell_x)]
        y_list = [(2 * i + 1) * self.mon.h_pixels / (2 * self.cell_y) for i in range(self.cell_y)]
        for i in x_list:
            for j in y_list:
                self.point_list.append([i,j])#where the points coordinates is 
        #print(self.point_list)
        self.point_list.remove(self.point_list[3]) # for screen recording 

        #######################################################
        #### prepare Kalman filters, R can change behaviour of Kalman filter
        #### play with it to get better smoothing, larger R - more smoothing and larger delay
        #######################################################

        # initialize Kalman filter for the on-screen gaze point-of regard
        self.kalman_filter_gaze = list()
        self.kalman_filter_gaze.append(Kalman1D(sz=1, R=0.01 ** 2))
        self.processors = []
        for cam_calib in cam_calibs:
            self.processors += [frame_processor(opt, cam_calib)]
        #temp
        # self.labels = cam_calib['face_g']
        # print(self.labels)
    
    def process(self, queues, gaze_network):
        # rets, imgs = [], []
        gaze_network.eval()
        frame_idx = 0
        break_mark = 0
        while break_mark == 0:
            tic = time.time() 

            imgs = []
            for queue in queues:
                imgs.append(queue.get()["frame"])
            ret_faces = []
            normalized_entries = {}
            for i, img in enumerate(imgs):
                ret_face, normalized_entry, patch_img = self.processors[i](img)
                ret_faces.append(ret_face)
                # print("normalized time: ", time.time() - tic)
                tic = time.time()
                #TODO:convert_input for muti-cam.
            # print(normalized_entries["gaze_cam"], normalized_entry)
                if np.all(ret_faces):
                    for key, value in normalized_entry.items():
                        add_kv(normalized_entries, key, value)
                    model_input = gaze_network.convert_input(normalized_entry)
                    output_dict = gaze_network(model_input)
                    # print("inference time: ", time.time() - tic)
                    # gaze_network.trans_totensor(normalized_entry)
                    self.calculate_pog_with_g(normalized_entry, output_dict)

                    if time.time() - self.start >  5 : 
                        self.start = time.time() # last point change time 
                        self.mark += 1  # mark which point should be shown

                    x_pixel_hat, y_pixel_hat = output_dict['PoG_px'][0], output_dict['PoG_px'][1]
                    # output_tracked = self.kalman_filter_gaze[0].update(x_pixel_hat + 1j * y_pixel_hat)
                    # x_pixel_hat, y_pixel_hat = np.ceil(np.real(output_tracked)), np.ceil(np.imag(output_tracked))
                    x_pixel_hat, y_pixel_hat = np.ceil(x_pixel_hat), np.ceil(y_pixel_hat)
                    x_pixel_hat = np.clip(x_pixel_hat, random.randint(0,50), random.randint(1870,1920))
                    y_pixel_hat = np.clip(y_pixel_hat, random.randint(0,50), random.randint(1030,1080))

                    display = np.ones((self.mon.h_pixels, self.mon.w_pixels, 3), np.float32)
                    cell_width = int(self.mon.w_pixels / self.cell_x)
                    cell_height = int(self.mon.h_pixels / self.cell_y)
                    for i in range(self.cell_y - 1):
                        cv.line(display,(0,(i+1)*cell_height),(self.mon.w_pixels,(i+1)*cell_height),(0, 0, 5), 5)
                    for i in range(self.cell_x - 1):
                        cv.line(display,((i+1)*cell_width,0),((i+1)*cell_width,self.mon.h_pixels),(0, 0, 5), 5)
                    point = self.point_list[self.mark%(self.cell_x * self.cell_y - 1)]
                    cv.circle(display,(int(point[0]),(int(point[1]))), 20, (0,0,0), -1)
                    cv.putText(display, '.', (int(x_pixel_hat), int(y_pixel_hat)), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 20, cv.LINE_AA)
                    cv.namedWindow("por", cv.WINDOW_NORMAL)
                    cv.setWindowProperty("por", cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)
                    if self.opt.display_patch:
                        img = draw_gaze(patch_img, output_dict['gaze'][0].detach().cpu().numpy() ,color = (255,0,100))
                        img = cv.flip(img, 1)
                        # img = draw_gaze(img, self.labels[frame_idx] ,color = (255,255,255))
                        h, w, c = img.shape
                        display[0:h, int(self.mon.w_pixels/2 - w/2):int(self.mon.w_pixels/2 + w/2), :] = 1.0 * img / 255.0
                    cv.imshow('por', display)
                    if cv.waitKey(1) & 0xFF == ord('q'):
                        cv.destroyAllWindows()
                        break_mark = 1
                        break
                frame_idx += 1
            imgs = []
            # if break_mark:
            #     for process in processes:
            #         process.join()
            # for cap in caps:
            #     ret, img = cap.read()
            #     rets.append(ret)
            #     imgs.append(img)


    def calculate_pog_with_g(self, input, output, output_gaze_key = "gaze", output_suffix = ''):
        
        
        origin = input['gaze_cam_origin']
        direction = output[output_gaze_key + output_suffix]
        if not isinstance(direction, np.ndarray):
            direction = direction.detach().cpu().numpy()[0]
        rotation = input['R']
        # print(origin.shape, direction.shape, rotation.shape)
        PoG_mm, PoG_px = to_screen_coordinates(origin, direction, rotation, self.mon)
        # print(origin.shape, PoG_mm.shape, PoG_px.shape)
        output['PoG_cm' + output_suffix] = 0.1 * PoG_mm
        output['PoG_px' + output_suffix] = PoG_px

        output['PoG_mm' + output_suffix] = \
            10.0 * output['PoG_cm' + output_suffix]



    def process_for_finetune(self, subject):
        num_image_per_point = self.opt.num_image_per_point
        data = {}
        with open('calibration/%s/%s/calib_target.pkl' %(self.opt.id, subject), 'rb') as f:
            targets = pickle.load(f)
        k = self.opt.k
        for i, processor in enumerate(self.processors):
            imgs_path = 'calibration/%s/%s/cam%d'% (self.opt.id, subject, i)
            if self.opt.visualize_cal:
                imgs_visual_path = 'calibration/%s/%s/calib_visual/cam%d'% (self.opt.id,subject, i)
                os.makedirs(imgs_visual_path, exist_ok=True)
            imgs_list = os.listdir(imgs_path)
            imgs_list.sort()

            train_indices = []
            for i in range(0, k*num_image_per_point, num_image_per_point):
                train_indices.append(random.sample(range(i, i + num_image_per_point), 3))
            train_indices = sum(train_indices, [])

            valid_indices = []
            for i in range(k*num_image_per_point, len(imgs_list) - num_image_per_point, num_image_per_point):
                valid_indices.append(random.sample(range(i, i + num_image_per_point), 1))
            valid_indices = sum(valid_indices, [])
            indices = train_indices + valid_indices
            for k, img_file in enumerate(imgs_list):
                if k not in indices:
                    continue
                img = cv.imread(os.path.join(imgs_path,img_file))
                target_3d = targets[k]

                ret_face, normalized_entry, patch_img = processor(img, target_3d)
                if ret_face:
                    if self.opt.visualize_cal:
                        img = draw_gaze(patch_img, normalized_entry['normalized_gaze'] ,color = (255,0,100))
                        cv.imwrite(os.path.join(imgs_visual_path,img_file), img)
                # add all imgs to data    
                    for key, value in normalized_entry.items():
                        add_kv(data, key, value)
        return data


