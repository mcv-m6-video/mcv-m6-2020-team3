from mrcnn.config import Config
from mrcnn import utils
import glob
import cv2
import utils_w3 as utw3
from sklearn.model_selection import train_test_split

import numpy as np

class AICityDataset(utils.Dataset):
    def get_Images(self, height, width, framePath, length=None, isTrain = True, trainSplit = 0.25, method="first"):
        gtPath = 'Datasets/AICity/aicity_annotations.xml'
        groundTruth = utw3.read_annotations(gtPath, length)
        groupedGT = [[] for x in range(length)]
        for elem in groundTruth:
            currFrame = elem['frame'] 
            groupedGT[currFrame].append(elem)
        self.groundTruth = groupedGT
        self.add_class("AICity", 1, "Car")
        framePaths = glob.glob(framePath + '/*.jpg')
        framePaths = sorted(framePaths)
        framePaths = framePaths[0:length] if length is not None else framePaths
        splitPoint = int(length*trainSplit)
        datasetFrames = framePaths[0:splitPoint] if isTrain is True else framePaths[splitPoint:]
        for i, frame in enumerate(datasetFrames):
            self.add_image("AICity", image_id=i, path=frame, width=width, height=height)
    
    def load_image(self, image_id):
        info = self.image_info[image_id]
        return cv2.imread(info['path'])





