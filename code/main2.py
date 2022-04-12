# coding=utf-8
#设计思路：调用相机摄像头，读取一帧图片，然后检测关键点，提取输入部分，输入到模型中，模型给出输出，用#来刷新界面上的点，然后刷新界面
from time import time
import numpy as np
import cv2
import os 
os.makedirs("frames", exist_ok = True)
#cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
cap0 = cv2.VideoCapture(0)
cap1 = cv2.VideoCapture(4)
cap2 = cv2.VideoCapture(6)


cap1.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
cap2.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
cap0.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))


#cap = cv2.VideoCapture(0) #参数为0时调用本地摄像头；url连接调取网络摄像头；文件地址获取本地视频
cap0.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap0.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap1.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
cap2.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
i = 0
while(True):
    tic = time()
    ret,frame0=cap0.read()
    ret,frame1=cap1.read()
    ret,frame2=cap2.read()
    print(time() - tic)
    i += 1
#灰度化
    #gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    #cv2.imshow('frame',gray)
    frame1 = cv2.resize(frame1, (640, 480))
    frame2 = cv2.resize(frame2, (640, 480))
    frame0 = cv2.resize(frame0, (640, 480))


    frame = np.hstack((frame1, frame2, frame0))
#普通图片
    cv2.imshow('frame',frame)

    cv2.imwrite(os.path.join("frames","%05d.png"%i), frame)
    if cv2.waitKey(1)&0xFF==ord('q'):
        break
cap1.release()
cap2.release()
cap0.release()

cv2.destroyAllWindows()
