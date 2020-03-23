from itertools import product
import numpy as np
import os
import pickle
import cv2
# Optical flow
#from block_matching import block_matching_optical_flow
from opt_flow import coarse2Fine, farneback, lucas_kanade, read_file
from optical_flow_functions import calculate_msen, read_Opflow, calculate_error, calculate_pepn, plot_Opflow_error
from Plot_OF import  read_opticalflow, read_sequences
from utils_w4 import create_folder

#sequences_path = '../Datasets/data_stereo_flow/training/image_0/'
#sequences_path = '../Datasets/cars/images/train/'
gt_path = '../Datasets/data_stereo_flow/training/flow_noc/'
gt_noc = "../Datasets/data_stereo_flow/training/flow_noc/000045_10.png"

opticalflow_path = '../Datasets/results_opticalflow_kitti/results/'
test = "../Datasets/results_opticalflow_kitti/results/LKflow_000045_10.png"


frame1_rgb = cv2.imread('../Datasets/data_stereo_flow/training/image_0/000045_10.png')
frame2_rgb = cv2.imread('../Datasets/data_stereo_flow/training/image_0/000045_11.png')

save_path = 'plots/'
create_folder(save_path + 'BM_optflow/')
create_folder(save_path + 'GT_optflow/')
create_folder(save_path + 'results/')

flow_gt, flow_test = read_Opflow(gt_noc, test)
gt = read_file(test)
# PyFlow
flow, flow_return = coarse2Fine(frame1_rgb, frame2_rgb, True)
# pyflow = coarse2Fine(sequences[0], sequences[1], True)
#
vect_err = calculate_error(flow_return, flow_gt)

msen = calculate_msen(flow_return, flow_gt)
pepn = calculate_pepn(flow_return, flow_gt, threshold=3)
plot_Opflow_error(flow_return, flow_gt, bins=100)

print("MSEN = {}".format(msen))
print("PEPN = {}".format(pepn))


# Farneback
farneback = farneback(frame1_rgb, frame2_rgb)
#farneback = farneback(sequences[0], sequences[1])
#msen, pepn = calculate_msen(flow_gt, farneback)
vect_err = calculate_error(farneback, flow_gt)

msen = calculate_msen(farneback, flow_gt)
pepn = calculate_pepn(farneback, flow_gt, threshold=3)
plot_Opflow_error(farneback, flow_gt, bins=100)

print("MSEN = {}".format(msen))
print("PEPN = {}".format(pepn))

# Lucas kanade
lucas_kanade = lucas_kanade(frame1_rgb, frame2_rgb)
# lucas_kanade = lucas_kanade(sequences[0], sequences[1])
#msen, pepn = calculate_msen(flow_gt, lucas_kanade)
vect_err = calculate_error(lucas_kanade, flow_gt)

msen = calculate_msen(lucas_kanade, flow_gt)
pepn = calculate_pepn(lucas_kanade, flow_gt, threshold=3)
plot_Opflow_error(lucas_kanade, flow_gt, bins=100)

print("MSEN = {}".format(msen))
print("PEPN = {}".format(pepn))