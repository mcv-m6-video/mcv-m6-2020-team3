import os
import numpy as np
import sys
import glob
import random
import tensorflow as tf
import matplotlib.pyplot as plt
import skimage
import cv2
import pickle
import utils as ut
from utils_Gaussian import read_video_and_divide, calculate_mean_std_first_part_video, calculate_mask, find_detections
import utils as ut
from tqdm import tqdm
from utils_read import read_gt_txt, transform_gt
from utils import addBboxesToFrames_avi, calculate_mAP, bb_iou, addBboxesToFrames_gif, upscaleDetections
from collections import deque
from utilities.BoundingBox import *
import week2.t3 as t3
from moviepy.editor import VideoFileClip
from utilities.VehicleDetector import VehicleDetector
#####################################################
from AICityIterator import AICityIterator
from AICityIterator import getStructure
# Root directory of the project
from week5.utilities.BoundingBox import draw_box_label

ROOT_DIR = os.path.abspath("../Mask_RCNN")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library


class VehicleDetections:
    def __init__(self, min_conf=0.6, max_age=4, max_hits=10):
        # Initialize constants
        self.max_age = max_age                   # no. of consecutive unmatched detection before a track is deleted
        self.min_hits = max_hits                 # no. of consecutive matches needed to establish a track
        self.vehicle_detected = False
        self.count = 0

        # Set up 'Vehicle Detector'
        self.detector = VehicleDetector(kitti=False, min_conf=min_conf)

testingSequences = ['S03']
vdt = VehicleDetections(min_conf=0.8, max_age=2, max_hits=8)
AICityStructure = getStructure()
ch_color = False
space = "gray"
showme = True
typeAlg = 'KNN'
backSub = t3.chooseAlgorithm(typeAlg)
test_path = "../Datasets/AIC20_track3/train/S03/"
size = (1920, 1080)
fps = 10
for seq in testingSequences:
    for cam in AICityStructure[seq]:
        print(cam)
        name = cam + "_detections_technique3"
        if not os.path.exists(name):
            os.makedirs(name)
        iterator = AICityIterator(seq, cam)
        pklList = []
        for frame, imgPath in tqdm(enumerate(iterator), total=len(iterator)):
            image = cv2.imread(imgPath)
            image_rgb= image
            if ch_color:
                im_change = ut.chage_color_space(image, space)
                if (space == "gray"):
                    image = im_change

                else:
                    image = im_change[:, :, 2]

            # if (showme):
            #     cv2.imshow('hsv_img', im_change)
            # fgMask = backSub.apply(image[:, :, 0])
            # cv2.imshow('FG Mask_Antes', fgMask)
            # fgMask_out = cv2.resize(fgMask ,image.shape[1::-1])
            # img2 = np.zeros_like(image_rgb)
            # img2[:, :, 0] = fgMask_out/255
            # img2[:, :, 1] = fgMask_out/255
            # img2[:, :, 2] = fgMask_out/255
            #
            # imagemulty  = cv2.multiply(img2, image_rgb)
            # cv2.imshow('imagemulty', imagemulty)

            dims = image.shape[:2]
            vdt.count += 1
            # Get bounding boxes for located vehicles

            det_boxes  = vdt.detector.get_bounding_box_locations(image_rgb, frame,pklList, name, frame)

        with open('detections/detections_{}_{}.pkl'.format(seq, cam), "wb") as f:
            pickle.dump(pklList, f)

        gt = read_gt_txt('{}{}/gt/gt.txt'.format(test_path, cam))
        tracks_gt_list = transform_gt(gt)
        print("calculate mAP...")
        mAP = calculate_mAP(gt, pklList, IoU_threshold=0.5, have_confidence=True, verbose=True)

# print("Images with GT and predictions in gt_" + str(image_id_predict) + ".png and pred_" + str(image_id_predict) + ".png")
