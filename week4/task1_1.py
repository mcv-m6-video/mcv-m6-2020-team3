import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from metrics import flow_metrics
from math import sqrt
import time


matching_methods = ['cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']


def read_file(path):
    """Read an optical flow map from disk
    Optical flow maps are stored in disk as 3-channel uint16 PNG images,
    following the method described in the KITTI optical flow dataset 2012
    (http://www.cvlibs.net/datasets/kitti/eval_stereo_flow.php?benchmark=flow).
    Returns:
      numpy array with shape [height, width, 3]. The first and second channels
      denote the corresponding optical flow 2D vector (u, v). The third channel
      is a mask denoting if an optical flow 2D vector exists for that pixel.
      Vector components u and v values range [-512..512].
    """
    data = cv2.imread(path, -1).astype('float32')
    result = np.empty(data.shape, dtype='float32')
    result[:,:,0] = (data[:,:,2] - 2**15) / 64
    result[:,:,1] = (data[:,:,1] - 2**15) / 64
    result[:,:,2] = data[:,:,0]

    return result


def block_matching(frame1, frame2, block_size=16, search_area=64, method='cv2.TM_CCORR_NORMED', bwd=False):
    height = frame1.shape[1]
    width = frame1.shape[0]
    if method == 'cv2.TM_CCORR_NORMED':
        threshold = 0.9997
    elif method == 'cv2.TM_CCOEFF_NORMED':
        threshold = 0.997
    else:
        threshold = 99999999999

    if bwd:
        temp = frame1
        frame1 = frame2
        frame2 = temp

    search_area = int(search_area/2)
    motion_blocks = []  # for plotting purposes
    motion = np.zeros([frame1.shape[0], frame1.shape[1], 3])
    for i in range(0, width, block_size):
        for j in range(0, height, block_size):
            center_i = int(round(i+(block_size/2)))  # center i of the block
            center_j = int(round(j+(block_size/2)))  # center j of the block
            block = frame1[i:i+block_size, j:j+block_size]
            # search area in i goes from center_i-search_area/2 (if > 0) to center_i+search_area/2, and the same for j
            search_space = frame2[max(0, center_i-search_area):min(center_i+search_area, frame1.shape[0]),
                                  max(0, center_j-search_area):min(center_j+search_area, frame1.shape[1])]

            meth = eval(method)
            res = cv2.matchTemplate(search_space, block, meth)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)  # max_loc gives the upper left corner of the sliding window

            if max_val < threshold:
                cent_i = cent_j = int(search_area-(block_size/2))  # upper left corner of the block in the search area
                # checks in case the upper left corner of the block is displaced (in the borders)
                if center_i - search_area < 0:
                    cent_i = cent_i + (center_i - search_area)
                if center_j - search_area < 0:
                    cent_j = cent_j + (center_j - search_area)

                if not bwd:  # distance from the highest response to the center of the search space
                    displacement = np.array(np.array(max_loc) - [cent_j, cent_i])
                else:
                    displacement = np.array([cent_j, cent_i]) - np.array(max_loc)
            else:
                displacement = [0, 0]
            motion_blocks.append(([center_i, center_j], [displacement[1], displacement[0]]))
            motion[i:i+block_size, j:j+block_size] = np.array([displacement[0], displacement[1], 1])

    return motion_blocks, motion


def main():

    # frame1_rgb = cv2.imread('000045_10.png')
    # frame2_rgb = cv2.imread('000045_11.png')
    # gt = read_file('gt_000045_10.png')
    # frame1_rgb = cv2.imread('/home/sanket/Documents/mcv-m6-2020-team3/Datasets/data_stereo_flow/training/image_0/000157_10.png')
    # frame2_rgb = cv2.imread('/home/sanket/Documents/mcv-m6-2020-team3/Datasets/data_stereo_flow/training/image_0/000157_11.png')
    # gt = read_file('/home/sanket/Documents/mcv-m6-2020-team3/Datasets/data_stereo_flow/training/flow_noc/000157_10.png')

    frame1_rgb = cv2.imread('/home/sanket/Documents/mcv-m6-2020-team3/Datasets/data_stereo_flow/training/image_0/000045_10.png')
    frame2_rgb = cv2.imread('/home/sanket/Documents/mcv-m6-2020-team3/Datasets/data_stereo_flow/training/image_0/000045_11.png')
    gt = read_file('/home/sanket/Documents/mcv-m6-2020-team3/Datasets/data_stereo_flow/training/flow_noc/000045_10.png')

    frame1 = cv2.cvtColor(frame1_rgb, cv2.COLOR_RGB2GRAY)
    frame2 = cv2.cvtColor(frame2_rgb, cv2.COLOR_RGB2GRAY)
    MSEN = np.zeros((15,15))
    PEPN = np.zeros((15,15))
    i = 0
    for block_size in [16]:  # search space
        j=0
        for search_area in [32]:
            if search_area <= block_size: continue
            start = time.time()
            motion_blocks, motion = block_matching(frame1, frame2, block_size=block_size, search_area=search_area, bwd=False)
            end = time.time()
            print(end - start)
            msen, pepn, img_err, vect_err = flow_metrics(motion.astype(np.float32), gt)
            print("Search area ", search_area, "block size", block_size, "MSEN: ", msen, " - PEPN: ", pepn)
            MSEN[i, j] = msen
            PEPN[i, j] = pepn
            j=j+1
            # PEPN.append(pepn)
        i = i+1
    # Plot the surface.
    # fig = plt.figure()
    # ax = fig.gca(projection='3d')
    #
    #
    # x = np.arange(8, 68, 4)
    # y = np.arange(8, 68, 4)
    # x, y = np.meshgrid(x, y)
    # surf = ax.plot_surface(x, y, MSEN, cmap='afmhot',
    #                        linewidth=0, antialiased=False)
    #
    # # Add a color bar which maps values to colors.
    # fig.colorbar(surf, shrink=0.5, aspect=5)
    # ax.set_title('MSEN')
    # ax.set_xlabel('Block size')
    # ax.set_ylabel('Search area')
    # ax.set_zlabel('MSEN error')
    #
    # plt.show()

    plt.imshow(img_err)
    plt.show()

    motion_module = np.sqrt(motion[:, :, 1] ** 2 + motion[:, :, 0] ** 2)
    plt.imshow(motion_module)
    plt.title("module of the motion vectors")
    plt.show()

    max_motion_module = np.max(motion_module)
    min_motion_module = np.min(motion_module)
    for motion_block in motion_blocks:
        center = motion_block[0]
        displacement = motion_block[1]
        green = 255 * ((sqrt(displacement[0]**2 + displacement[1]**2) - min_motion_module) /
                       (max_motion_module - min_motion_module))*1
        cv2.arrowedLine(frame1_rgb, (center[1], center[0]),
                        (center[1]+displacement[1]*2, center[0]+displacement[0]*2), (0, green, 255), 2, tipLength=0.4)


    cv2.imshow("motion", frame1_rgb)
    cv2.waitKey()

    return


if __name__ == '__main__':
    main()
    # print(timeit.timeit(block_matching(), number=100000))
    # x = np.asarray([8, 12 , 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64])
    # y = np.asarray([8, 12 , 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64])
    # z = np.asarray(10.277124, 9.6373, 8.877609, 7.941295, 7.484724, 6.9695964, 6.611499, 6.452316, 6.338886, 6.2480536, 6.2353067, 6.407853, 6.559667,
    #                6.756943, 10.10727, 9.262248, 8.403364, 7.5238, 6.936257, 6.3451214, 6.0333867,              )
    # z1 = np.asarray()
