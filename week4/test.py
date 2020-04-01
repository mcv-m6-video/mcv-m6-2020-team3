#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 18:12:05 2020

@author: kaiali
"""


import cv2
from videostab import VideoStabilizer

video = cv2.VideoCapture("test.mp4")
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out2 = cv2.VideoWriter('out5.avi', fourcc, 15.0, (640,480))
#out3 = cv2.VideoWriter('out6.avi', fourcc, 15.0, (640,480))
stabilizer = VideoStabilizer(video)

while True:
        ret,frame2=video.read()
        out2.write(frame2)
        cv2.imshow("frame2", frame2)
        success, _, frame = stabilizer.read()
        out2.write(frame)
        if not success:
                print("No frame is captured.")
                break
        cv2.imshow("frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
                break