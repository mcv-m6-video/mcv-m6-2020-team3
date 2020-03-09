from __future__ import print_function
import cv2 as cv
import os
import sys
import utils
sys.path.insert(0, "/python/path/to/DEBUG/installation/of/opencv")
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse
#Task3: Compare with state - of - the - art


def chooseAlgorithm(typeBS):
    if typeBS == 'MOG':
        backSub = cv.bgsegm.createBackgroundSubtractorMOG()
    elif typeBS == 'MOG2':
        backSub = cv.createBackgroundSubtractorMOG2()
    elif typeBS == 'MOG2_notshadows':
        backSub = cv.createBackgroundSubtractorMOG2(detectShadows=False)
    elif typeBS == 'KNN':
        backSub = cv.createBackgroundSubtractorKNN()
    elif typeBS == 'LSBP':
        backSub = cv.createBackgroundSubtractorLSBP()
    elif typeBS == 'GMG':
        backSub = cv.bgsegm.createBackgroundSubtractorGMG()
    return backSub

def video_input (path, inputtype):
    if inputtype == 'video':
        capture = cv.VideoCapture(cv.samples.findFileOrKeep(path))

    return capture





