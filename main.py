from __future__ import print_function
import cv2 as cv
import os
import sys
import utils
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse
import t3

if __name__ == "__main__":
    # Read groundtruth
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    path = ROOT_DIR+'/Datasets/AICity/frames/'
    typeAlg = 'MOG2_notshadows'
    inputtype = 'sequence'
    #numeberofimages = 150 #(nuumber of images)
    numeberofimages = 2141
    save_path = ROOT_DIR + '/output_frames/' + typeAlg + '/'
    print("Reading annotations...")
    backSub = t3.chooseAlgorithm(typeAlg)
    verbose = False
    IoU_mean_list = []
    mAP_mean_list = []
    detections = []
    pbar = tqdm(total=numeberofimages)
    iousPerFrame_list = []
    mAP_list = []
    detections = []
    showme = False
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for x in range(1, numeberofimages):

        # read images
        pathim = path+'%05d.jpg' % x
        print(pathim)
        frame = cv.imread(pathim)
        #TODO (task 4 -GABY -KEYAO)
        #pre- processim (change space like YUV or lab)

        #TODO (TASK 1  Gaussian modelling - YIXIONG)
        #IMPUT IMAGE (TRY GRY SCALE AND COLOR OPTION) OUTPUT MASK

        #TODO (TASK 2.1  Adaptive modelling - Obtain first the best 𝛼 for non-recursive, and later estimate ⍴ for the recursive cases- SANKEY)
        #IMPUT IMAGE (TRY GRY SCALE AND COLOR OPTION) OUTPUT MASK

        #TODO (TASK 2.2  Adaptive modelling - Optimize (𝛼, ⍴) together with grid search or random search - MARC)
        # IMPUT IMAGE (TRY GRY SCALE AND COLOR OPTION) OUTPUT MASK

        #TODO (TASK 3  Comparison with state - of - the - art - GABY)
        fgMask = backSub.apply(frame)
        fgmask_out = cv.resize(fgMask, (0, 0), fx=0.3, fy=0.3)
        cv.imwrite(save_path+'frame_' + '%05d.jpg' % x, fgmask_out.astype('uint8') * 255)
        if (showme):
            cv.imshow('Frame', frame)
            cv.imshow('FG Mask_Antes', fgMask)
        keyboard = cv.waitKey(30)
        #TODO (TASK 4  Post Processing - GABY )
        maskMOG2 = utils.morphological_filtering(fgMask)
        if (showme):
            cv.imshow('FG Mask', maskMOG2)
        detections = utils.candidate_window(showme, save_path, frame, x, maskMOG2, detections)
        pbar.update(1)
        keyboard = cv.waitKey(30)
        if keyboard == 'q' or keyboard == 27:
            break
    cv.destroyAllWindows()
    pbar.close()

    # TODO ALL TASKS - Evaluattion GABY -KEYAO
    groundTruth = utils.read_annotations(ROOT_DIR + '/Datasets/AICity/aicity_annotations.xml', numeberofimages)
    groundTruthPerFrame = utils.getDetectionsPerFrame(groundTruth)
    #detectionsMOG_filtered = [x for x in detections if x.frame > int(2141 * 0.25)]
    mAP = utils.calculate_mAP(groundTruth, detections, IoU_threshold=0.5, have_confidence=False)
