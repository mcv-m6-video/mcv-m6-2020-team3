from __future__ import print_function
import cv2
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
        backSub = cv2.bgsegm.createBackgroundSubtractorMOG()
    elif typeBS == 'MOG2':
        backSub = cv2.createBackgroundSubtractorMOG2()
    elif typeBS == 'MOG2_notshadows':
        backSub = cv2.createBackgroundSubtractorMOG2(detectShadows=False)
    elif typeBS == 'KNN':
        backSub = cv2.createBackgroundSubtractorKNN()
    elif typeBS == 'CNT':
        backSub = cv2.bgsegm.createBackgroundSubtractorCNT()
    elif typeBS == 'GMG':
        backSub = cv2.bgsegm.createBackgroundSubtractorGMG()
    return backSub
