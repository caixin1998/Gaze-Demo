# coding=utf-8
#设计思路：调用相机摄像头，读取一帧图片，然后检测关键点，提取输入部分，输入到模型中，模型给出输出，用#来刷新界面上的点，然后刷新界面
from time import time
import numpy as np
import cv2
import os 
os.makedirs("frames", exist_ok = True)
#cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
cap = cv2.VideoCapture(4)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
#cap = cv2.VideoCapture(0) #参数为0时调用本地摄像头；url连接调取网络摄像头；文件地址获取本地视频
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
i = 0
while(True):
    tic = time()
    ret,frame=cap.read()
    frame = cv2.flip(frame, -1)
    print(time() - tic)

    i += 1
#灰度化
    #gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    #cv2.imshow('frame',gray)

#普通图片
    cv2.imshow('frame',frame)
    key = cv2.waitKey(1000)
    # cv2.imwrite(os.path.join("frames","%05d.png"%i), frame)
    if key&0xFF==ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
