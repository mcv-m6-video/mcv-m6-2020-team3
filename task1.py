import utils
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils_Gaussian import Gaussian_modelling


if __name__ == "__main__":
    # too long will need a lot of time, so decrease
    video_length = 2141
    # video_length = 100
    video_split_ratio = 0.25
    video_path = "./Datasets/AICity_data/train/S03/c010/vdo.avi"
    groundtruth_xml_path = "./Datasets/aicity_annotations.xml"
    # groundtruth_path = "../datasets/AICity_data/train/S03/c010/gt/gt.txt"
    roi_path = 'Datasets/AICity_data/train\S03/c010/roi.jpg'

    print("Reading annotations...")
    groundTruth = utils.read_annotations('Datasets/AICity/aicity_annotations.xml', video_length)
    groundTruthPerFrame = utils.getDetectionsPerFrame(groundTruth)
    gt_filtered = [x for x in groundTruth if x['frame'] > int(video_length * video_split_ratio)]

    # This function lasts about 10 minutes
    foreground_second_part, detections = Gaussian_modelling(roi_path, video_path, alpha=1.25, rho=1,
                                                            video_length=video_length,
                                                            video_split_ratio = video_split_ratio)
    mAP = utils.calculate_mAP(groundTruth, detections, IoU_threshold=0.5, have_confidence=False)

    print(mAP)

pass
