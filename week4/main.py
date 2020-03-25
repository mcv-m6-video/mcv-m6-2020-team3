import cv2
# Optical flow
from opt_flow import coarse2Fine, farneback, lucas_kanade_denso, read_file
from optical_flow_functions import calculate_msen, read_Opflow, calculate_error, calculate_pepn, plot_Opflow_error

gt_noc = "../Datasets/data_stereo_flow/training/flow_noc/000045_10.png"

test = "../Datasets/results_opticalflow_kitti/results/LKflow_000045_10.png"


frame1_rgb = cv2.imread('../Datasets/data_stereo_flow/training/image_0/000045_10.png')
frame2_rgb = cv2.imread('../Datasets/data_stereo_flow/training/image_0/000045_11.png')
save_path = 'plots/'

flow_gt, flow_test = read_Opflow(gt_noc, test)

# PyFlow
flow, flow_return = coarse2Fine(frame1_rgb, frame2_rgb, True)
vect_err = calculate_error(flow_return, flow_gt)
msen = calculate_msen(flow_return, flow_gt)
pepn = calculate_pepn(flow_return, flow_gt, threshold=3)
plot_Opflow_error(flow_return, flow_gt, bins=100)
print("MSEN = {}".format(msen))
print("PEPN = {}".format(pepn))

# Farneback
farneback = farneback(frame1_rgb, frame2_rgb, True)
vect_err = calculate_error(farneback, flow_gt)
msen = calculate_msen(farneback, flow_gt)
pepn = calculate_pepn(farneback, flow_gt, threshold=3)
plot_Opflow_error(farneback, flow_gt, bins=100)
print("MSEN = {}".format(msen))
print("PEPN = {}".format(pepn))

# Lucas kanade
flow = lucas_kanade_denso(frame1_rgb, frame2_rgb, True)
vect_err = calculate_error(flow, flow_gt)
msen = calculate_msen(flow, flow_gt)
pepn = calculate_pepn(flow, flow_gt, threshold=3)
plot_Opflow_error(flow, flow_gt, bins=100)
print("MSEN = {}".format(msen))
print("PEPN = {}".format(pepn))
