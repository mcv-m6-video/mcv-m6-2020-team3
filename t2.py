import cv2
import matplotlib.pyplot as plt 
import numpy as np 
import os

from utils import

def adaptive_modelling_background(video_path, alpha_factor, rho_factor, path_to_save, color_space):

	capture_frames = cv2.videoCapture(video_path)
	num_frames = capture_frames.get(cv2.CAP_PROP_FRAME_COUNT)
	frame_width = capture_frames.get(cv2.CAP_PROP_FRAME_WIDTH)
	frame_height = capture_frames.get(cv2.CAP_PROP_FRAME_HEIGHT)

	# Configuration for Training frames
    flag = True
	training_frames = num_frames * 0.25

	if colorspace == 'GRAYSCALE':
        frames_acc = np.zeros((height, width), dtype='float')
        var_acc = np.zeros((height, width), dtype='float')
    elif colorspace == 'YUV':
        frames_acc = np.zeros((height, width, 3), dtype='float')
        var_acc = np.zeros((height, width, 3), dtype='float')
    else:
        print("wrong color space")
        return


    iter_frame = 0

    while flag and iter_frame<training_frames:

    	flag, frames = capture_frames.read()
    	iter_frame = int(capture_frames.get(cv2.CAP_PROP_POS_FRAMES))

    	if colorspace == 'GRAYSCALE':
            frames = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        elif colorspace == 'YUV':
            frames = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
        frames_acc += frames

        # Mean and variance computation
        print('Mean and variance computation...')
        mean_image = frames_acc / training_frames

        # set the current frame to 0 and computing the variance

        capture_frames.set(cv2.CAP_PROP_POS_FRAMES, 0)

        iter_frame = int(capture_frames.get(cv2.CAP_PROP_POS_FRAMES))
        flag = True

        while flag and iter_frame<training_frames:
        	



































