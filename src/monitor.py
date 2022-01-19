#!/usr/bin/env python3

# --------------------------------------------------------
# Copyright (C) 2020 NVIDIA Corporation. All rights reserved.
# NVIDIA Source Code License (1-Way Commercial)
# Code written by Shalini De Mello.
# --------------------------------------------------------

import gi.repository
gi.require_version('Gdk', '3.0')
from gi.repository import Gdk
import numpy as np

class monitor:

    def __init__(self):
        display = Gdk.Display.get_default()
        screen = display.get_default_screen()
        default_screen = screen.get_default()
        num = default_screen.get_number()

        self.h_mm = default_screen.get_monitor_height_mm(num)
        self.w_mm = default_screen.get_monitor_width_mm(num)
        self.h_pixels = 1080
        self.w_pixels = 1920

        self.pixels_per_millimeter = (self.w_pixels / self.w_mm,  self.h_pixels / self.h_mm)

        self.inv_camera_transformation = np.eye(4)
        self.inv_camera_transformation[:3,3] = np.array([271, -24, 0])
        self.inv_camera_transformation[0][0] = -1
        self.inv_camera_transformation[2][2] = -1
        self.camera_transformation = np.linalg.pinv(self.inv_camera_transformation)
        self.camera_transformation[3] = np.array([0,0,0,1])


    def monitor_to_camera(self, x_pixel, y_pixel):

        # assumes in-build laptop camera, located centered and 10 mm above display
        # update this function for you camera and monitor using: https://github.com/computer-vision/takahashi2012cvpr
        pog_screen = np.array([x_pixel * self.w_mm / self.w_pixels, y_pixel * self.h_mm / self.h_pixels, 0, 1]).T
        pog_cam = np.dot(self.camera_transformation, pog_screen)
        return pog_cam[:3]

if __name__ == "__main__":
    mon = monitor()
    print(mon.h_mm)
    print(mon.w_mm)
    # print(mon.h_mm)
    print(mon.pixels_per_millimeter)
    print(mon.h_pixels)
    print(mon.w_pixels)
    print(mon.inv_camera_transformation)
    print(mon.camera_transformation)
    print(mon.monitor_to_camera(50,50))




    