import cv2
import os
import numpy as np
from matplotlib import pyplot as plt

"""
https://github.com/simonmeister/motion-rcnn/tree/master/devkit
Optical flow maps are saved as 3-channel uint16 PNG images: The first channel
contains the u-component, the second channel the v-component and the third
channel denotes if the pixel is valid or not (1 if true, 0 otherwise). To convert
the u-/v-flow into floating point values, convert the value to float, subtract
2^15 and divide the result by 64.0:

flow_u(u,v) = ((float)I(u,v,1)-2^15)/64.0;
flow_v(u,v) = ((float)I(u,v,2)-2^15)/64.0;
valid(u,v)  = (bool)I(u,v,3);
"""
def read_Opflow(estimation, gt):
    img_estimation = cv2.imread(estimation, -1)
    img_gt = cv2.imread(gt, -1)

    if (img_estimation.shape[0] != img_gt.shape[0]) or (img_estimation.shape[1] != img_gt.shape[1]):
        print("ERROR: file size is wrong!")
        return
    else:
        fu_estimation = (img_estimation[:, :, 2] - 2. ** 15) / 64
        fv_estimation = (img_estimation[:, :, 1] - 2. ** 15) / 64
        valid_estimation = img_estimation[:,:,0]
        Opflow_estimation = np.transpose(np.array([fu_estimation, fv_estimation, valid_estimation]), (1, 2, 0))

        fu_gt = (img_gt[:, :, 2] - 2. ** 15) / 64
        fv_gt = (img_gt[:, :, 1] - 2. ** 15) / 64
        valid_gt = img_gt[:,:,0]
        Opflow_gt = np.transpose(np.array([fu_gt, fv_gt, valid_gt]),(1,2,0))

        print(Opflow_estimation.shape)
        plt.figure(1)
        plt.imshow(Opflow_estimation)
        if not os.path.exists('results'):
            os.makedirs('results')
        plt.savefig('results/Opflow_estimation.png')
        plt.show()

        plt.figure(2)
        plt.imshow(Opflow_gt)
        plt.savefig('results/Opflow_gt.png')
        plt.show()
        
        return Opflow_estimation, Opflow_gt
def calculate_error(Opflow_test, Opflow_gt):
    Opflow_u = Opflow_test[:, :, 0] - Opflow_gt[:, :, 0]
    Opflow_v = Opflow_test[:, :, 1] - Opflow_gt[:, :, 1]
    Opflow_error = np.sqrt(Opflow_u * Opflow_u + Opflow_v * Opflow_v)

    valid_gt = Opflow_gt[:, :, 2]

    Opflow_error[valid_gt == 0] = 0

    plt.figure(1)
    plt.imshow(Opflow_error, cmap="magma")
    plt.colorbar()
    plt.tick_params(axis='both', labelbottom=False, labelleft=False)
    plt.savefig('results/Opflow_error.png')
    plt.show()

    return Opflow_error
# def calculate_error(pred, gt):
#     flowExist = (gt[:, :, 2] == 1)
#     pred_flow = pred[flowExist]
#     gt_flow = gt[flowExist]
#     # print(flowExist.shape)
#     img_err = np.zeros(shape=gt[:, :, 1].shape)
#
#     err = gt_flow[:, :2] - pred_flow[:, :2]
#
#     squared_err = np.sum(err ** 2, axis=1)
#     vect_err = np.sqrt(squared_err)
#     hit = vect_err < 3.0
#     img_err[flowExist] = vect_err
#
#     msen = np.mean(vect_err)
#     pepn = 100 * (1 - np.mean(hit))
#     plot_Opflow_error (vect_err,gt[:, :, 2], 100)
#     return msen, pepn, img_err, vect_err

def calculate_msen(Opflow_error, Opflow_gt):
    valid_gt = Opflow_gt[:, :, 2]
    msen = np.mean(Opflow_error[valid_gt != 0])
    return msen

def calculate_pepn(Opflow_error, Opflow_gt, threshold=3):
    valid_gt = Opflow_gt[:, :, 2]
    pepn = (np.sum(Opflow_error[valid_gt != 0] > threshold)/len(Opflow_error[valid_gt != 0]))
    return pepn

def plot_Opflow_error(Opflow_error, Opflow_gt, bins = 20):
    valid_gt = Opflow_gt[:, :, 2]
    plt.figure()
    plt.hist(Opflow_error[valid_gt != 0], bins=bins, density=True)
    plt.title('Density of Optical Flow Error')
    plt.xlabel('Optical Flow error')
    plt.ylabel('The Percentage of Pixels')
    plt.savefig('results/Opflow_error_2.png')
    plt.show()

if __name__ == "__main__":
    estimation1 = "./Datasets/results_opticalflow_kitti/results/LKflow_000045_10.png"
    gt1 = "./Datasets/data_stereo_flow/training/flow_noc/000045_10.png"

    estimation2 = "./Datasets/results_opticalflow_kitti/results/LKflow_000157_10.png"
    gt2 = "./Datasets/data_stereo_flow/training/flow_noc/000157_10.png"

    Opflow_test, Opflow_gt  = read_Opflow(estimation1, gt1)
    Opflow_error = calculate_error(Opflow_test, Opflow_gt)

    msen = calculate_msen(Opflow_error, Opflow_gt)
    pepn = calculate_pepn(Opflow_error, Opflow_gt, threshold=3)
    plot_Opflow_error(Opflow_error, Opflow_gt, bins=100)

    print("MSEN = {}".format(msen))
    print("PEPN = {}".format(pepn))

