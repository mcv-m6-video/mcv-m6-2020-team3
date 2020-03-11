import utils
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils_Gaussian import Gaussian_modelling
import cv2
import pickle

from utils_Gaussian import read_video_and_divide, calculate_mean_std_first_part_video, calculate_mask, find_detections


if __name__ == "__main__":
    # too long will need a lot of time, so decrease
    video_length = 2141
    # video_length = 100
    video_split_ratio = 0.25
    # video_path = "./Datasets/AICity_data/train/S03/c010/vdo.avi"
    video_path = "./Datasets/AICity/frames/"
    groundtruth_xml_path = 'Datasets/AICity/aicity_annotations.xml'
    # groundtruth_path = "../datasets/AICity_data/train/S03/c010/gt/gt.txt"
    roi_path = 'Datasets/AICity_data/train/S03/c010/roi.jpg'



    # foreground_second_part, detections = Gaussian_modelling(roi_path, video_path, alpha=11.0, rho=1,
    #                                                         video_length=video_length,
    #                                                         video_split_ratio = video_split_ratio)

    # Gaussian_modelling
    roi = cv2.cvtColor(cv2.imread(roi_path), cv2.COLOR_BGR2GRAY)

    flag_training = False
    training_filename = 'training.pkl'
    if flag_training:
        print("Reading annotations...")
        groundTruth = utils.read_annotations(groundtruth_xml_path, video_length)
        gt_filtered = [x for x in groundTruth if x['frame'] > int(video_length * video_split_ratio)]

        # begin training
        video_first_part, video_second_part, divide_frame = \
            read_video_and_divide(video_path, video_length=video_length, video_split_ratio=video_split_ratio)

        video_first_part_mean, video_first_part_std = calculate_mean_std_first_part_video(video_first_part)
        # end training
        with open(training_filename, 'wb') as f:
            pickle.dump([video_second_part, divide_frame, video_first_part_mean,
                         video_first_part_std, gt_filtered], f)
            f.close()
    else:
        print("Reading pkl")
        with open(training_filename, 'rb') as p:
            video_second_part, divide_frame, video_first_part_mean, video_first_part_std, gt_filtered = pickle.load(p)
            p.close()

    print("Reading annotations...")
    groundTruth = utils.read_annotations('Datasets/ai_challenge_s03_c010-full_annotation.xml', video_length)
    gt_filtered = [x for x in groundTruth if x['frame'] > int(video_length * video_split_ratio)]

    mAP_list = []
    for i in range(3, 4):
        # here we can set whether to test all the 75% of video
        flag_test_part = True
        test_length = 100
        if flag_test_part:
            video_second_part = video_second_part[:test_length, :, :]
            gt_filtered = [x for x in gt_filtered if x['frame'] <= (divide_frame + 1 + test_length)]


        print('Extracting foreground...')
        alpha = float(i)
        foreground_second_part = calculate_mask(roi, video_second_part, video_first_part_mean,
                                                video_first_part_std, alpha)

        detections = find_detections(foreground_second_part, first_frame_id=divide_frame + 1)

        print('Finish Extracting foreground...foreground_second_part.shape = {}', foreground_second_part.shape)

        mAP = utils.calculate_mAP(gt_filtered, detections, IoU_threshold=0.5, have_confidence=False, verbose=True)

        mAP_list.append(mAP)

        utils.addBboxesToFrames(video_path, detections, gt_filtered, "test")

    print(mAP_list)
pass
