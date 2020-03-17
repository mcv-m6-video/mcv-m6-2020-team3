from mrcnn.config import Config
from mrcnn import utils
import glob
import cv2
from utils_tracking import read_tracking_annotations
from utils_w3 import tf_bbox2mask
from sklearn.model_selection import train_test_split

import numpy as np

class AICityDataset(utils.Dataset):
    def get_Images(self, height, width, framePath, length=None, isTrain = True, trainSplit = 0.25, method="first"):
        gtPath = 'Datasets/AICity/aicity_annotations.xml'
        groundTruth, _ = read_tracking_annotations(gtPath, length)
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
        frameIds = list(range(length))
        frameIds = frameIds[0:splitPoint] if isTrain is True else frameIds[splitPoint:]
        for id, frame in zip(frameIds, datasetFrames):
            self.add_image("AICity", image_id=id, path=frame, width=width, height=height)
    
    def load_image(self, image_id):
        info = self.image_info[image_id]
        return cv2.resize(cv2.imread(info['path']), (info['height'], info['width']))

    def load_mask(self, image_id):
        info = self.image_info[image_id]
        detections = self.groundTruth[image_id]
        nDetections = len(detections)
        cls_ids = np.zeros(nDetections)
        mask = np.zeros([info['height'], info['width'], nDetections], dtype=np.uint8)

        for i, detection in enumerate(detections):
            #All detections are cars
            cls_ids[i] = 1
            y1 = detection['top']
            x1 = detection['left']
            y2 = detection['top'] + detection['height']
            x2 = detection['left'] + detection['width']
            tmpMask = tf_bbox2mask(y1, x1, y2, x2, 1080, 1920)
            tmpMask = cv2.resize(tmpMask, (info['height'], info['width']))
            mask[:,:,i] = tmpMask
        return mask.astype(np.bool), cls_ids.astype(np.int32)

    def image_reference(self, image_id):
        info = self.image_info[image_id]
        return info['path']
        





