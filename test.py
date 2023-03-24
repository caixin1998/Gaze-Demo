
import time
import cv2 as cv
import gi.repository
gi.require_version('Gdk', '3.0')
from gi.repository import Gdk
# import numpy as np
# from person_calibration_video import collect_data, fine_tune
# from core import process_core
# from models import create_model
# import multiprocessing as mp
# import pyrealsense2 as rs
import time
img = cv.imread("1.png")
cv.imshow('image', img)
key_press = cv.waitKey(2000)
print("key_press", key_press)