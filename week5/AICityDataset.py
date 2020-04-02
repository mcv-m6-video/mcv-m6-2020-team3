import os
import sys

# Root directory of the project
ROOT_DIR = os.path.abspath("./Mask_RCNN")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library

from mrcnn.config import Config
from mrcnn import utils
import glob
import cv2
from utils import bbox2mask
import numpy as np

from AICityIterator import AICityIterator
from AICityGroundtruth import getGroundtruth

class AICityDataset(utils.Dataset):
    datasetStructure = {
        'S01': ['c001', 'c002', 'c003', 'c004', 'c005'],
        'S03': ['c010', 'c011', 'c012', 'c013', 'c014', 'c015'],
        'S04': ['c016', 'c018', 'c020', 'c022', 'c024', 'c026', 'c028', 'c030', 'c032', 'c034', 'c036', 'c038', 'c040','c017', 'c019', 'c021', 'c023', 'c025', 'c027', 'c029', 'c031', 'c033', 'c035', 'c037', 'c039']
    }

    trainSequences = ['S01', 'S04']
    testSequences = ['S03']
    
    def get_Images(self, height, width, isTrain = True):    
        self.add_class("AICity", 1, "Car")
        id = 0
        sequences = self.trainSequences if isTrain is True else self.testSequences
        for sequence in sequences:
            for camera in self.datasetStructure[sequence]:
                for i, image in enumerate(AICityIterator(sequence, camera)):
                    if i == 0:
                        imgMat = cv2.imread(image)
                        origHeight = imgMat.shape[0]
                        origWidth = imgMat.shape[1]
                    self.add_image("AICity",
                                    image_id=id,
                                    seq=sequence,
                                    cam=camera,
                                    frame=i+1,
                                    path=image,
                                    width=width,
                                    height=height,
                                    origHeight=origHeight,
                                    origWidth=origWidth)
                    id += 1
    
    def load_image(self, image_id):
        info = self.image_info[image_id]
        return cv2.resize(cv2.imread(info['path']), (info['height'], info['width']))

    def load_mask(self, image_id):
        info = self.image_info[image_id]
        origHeight = info['origHeight']
        origWidth = info['origWidth']
        detections = getGroundtruth(info['seq'], info['cam'], info['frame'])
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
            tmpMask = bbox2mask(y1, x1, y2, x2, origHeight, origWidth)
            tmpMask = cv2.resize(tmpMask, (info['height'], info['width']))
            mask[:,:,i] = tmpMask
        return mask.astype(np.bool), cls_ids.astype(np.int32)

    def image_reference(self, image_id):
        info = self.image_info[image_id]
        return info['path']
        





