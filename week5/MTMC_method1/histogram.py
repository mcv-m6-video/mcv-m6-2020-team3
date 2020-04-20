import numpy as np
import cv2
import sys
import os
from matplotlib import pyplot as plt


def compute_histogram(im, block_factor=3, color_space='HSV'):
    """
    Computes the histogram as grid of patches with size img.size/block_factor.
    Returns list of histograms of each block.
    The histograms are indexed in following way:
    hist[block_idx][channel]
    """

    # Shape = rows and columns
    remainder_rows = im.shape[0] % block_factor
    remainder_cols = im.shape[1] % block_factor

    im_block = cv2.copyMakeBorder(im, block_factor - remainder_rows, 0, block_factor - remainder_cols, 0,
                                  cv2.BORDER_CONSTANT)

    windowsize_r = int(im_block.shape[0] / block_factor)
    windowsize_c = int(im_block.shape[1] / block_factor)

    # print(im_block.shape)
    # print(str(windowsize_r)+' '+str(windowsize_c))
    # cv2.imshow("fullImg", im_block)

    hist = []
    for r in range(0, im_block.shape[0], windowsize_r):
        for c in range(0, im_block.shape[1], windowsize_c):
            hist_blocks = []
            window = im_block[r:r + windowsize_r, c:c + windowsize_c]
            if color_space == 'GRAY':
                window_gray = cv2.cvtColor(window, cv2.COLOR_BGR2GRAY)
                hist_block = cv2.calcHist([window_gray], [0], None, [256], [0, 256])
                cv2.normalize(hist_block, hist_block, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
                hist_blocks.append(hist_block)
            elif color_space == 'RGB':
                hist_block = cv2.calcHist([window], [0], None, [256], [0, 256])
                cv2.normalize(hist_block, hist_block, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
                hist_blocks.append(hist_block)
                hist_block = cv2.calcHist([window], [1], None, [256], [0, 256])
                cv2.normalize(hist_block, hist_block, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
                hist_blocks.append(hist_block)
                hist_block = cv2.calcHist([window], [2], None, [256], [0, 256])
                cv2.normalize(hist_block, hist_block, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
                hist_blocks.append(hist_block)
            elif color_space == 'HSV':
                window = cv2.cvtColor(window, cv2.COLOR_BGR2HSV)
                hist_block = cv2.calcHist([window], [0], None, [256], [0, 256])
                cv2.normalize(hist_block, hist_block, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
                hist_blocks.append(hist_block)
                hist_block = cv2.calcHist([window], [1], None, [256], [0, 256])
                cv2.normalize(hist_block, hist_block, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
                hist_blocks.append(hist_block)
                hist_block = cv2.calcHist([window], [2], None, [256], [0, 256])
                cv2.normalize(hist_block, hist_block, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
                hist_blocks.append(hist_block)

            hist.append(hist_blocks)

    return hist


def compare_histogram_blocks(hist1, hist2, method=1):
    """
    methods: 1=chi-square, 2=histogram-intersection 3=Hellinger-distance
    """
    if len(hist1) == 3:
        score1 = cv2.compareHist(hist1[0], hist2[0], method=method)
        score2 = cv2.compareHist(hist1[1], hist2[1], method=method)
        score3 = cv2.compareHist(hist1[2], hist2[2], method=method)
        score = score1 + score2 + score3
    else:
        score = cv2.compareHist(hist1[0], hist2[0], method=method)
    return score


if __name__ == "__main__":
    img = cv2.imread("../../images/segmented.png", 1)
    hist = compute_histogram(img, 4, 'GRAY')
    print(np.shape(hist))
    score = compare_histogram_blocks(hist[0], hist[8])
    print(score)
