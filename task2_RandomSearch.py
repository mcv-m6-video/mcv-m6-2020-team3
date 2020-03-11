import utils
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils_Gaussian import Gaussian_modelling
from AdaptiveBGModeling import AdaptiveBGModeling
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import ParameterSampler
import io
import sys


if __name__ == "__main__":
    # too long will need a lot of time, so decrease
    # video_length = 2141
    video_length = 800
    rSearchIterations = 10
    video_split_ratio = 0.25
    video_path = "./Datasets/AICity/frames/"
    groundtruth_xml_path = "./Datasets/aicity_annotations.xml"
    roi_path = 'Datasets/AICity/train/S03/c010/roi.jpg'

    print("Reading annotations...")
    groundTruth = utils.read_annotations('Datasets/AICity/aicity_annotations.xml', video_length)
    groundTruthPerFrame = utils.getDetectionsPerFrame(groundTruth)
    gt_filtered = [x for x in groundTruth if x['frame'] > int(video_length * video_split_ratio)]

    params = {}
    params['alpha'] = np.arange(0.25, 3, 0.05)
    #Uncomment this line when using for Adaptive Modeling (Also consider chaning the possible values for alpha and rho)
    #params['rho'] = np.arange(0.25, 3, 0,25)

    #SKLearn implementation of RandomSearch (uses multiples folds which doesn't apply for our case, longer computation)
    """model = AdaptiveBGModeling(roi_path, video_path, video_length, video_split_ratio, gt_filtered)
    rSearch = RandomizedSearchCV(model, params, 5, cv = 2)
    rSearch.fit(video_path)
    print(rSearch.score(None))
    print(rSearch.best_params_)"""

    #Generation of parameter candidates
    randomParameters = list(ParameterSampler(params, n_iter=rSearchIterations))
    bestIteration = 0
    bestScore = 0

    for i, combination in tqdm(enumerate(randomParameters), total=rSearchIterations):
        print("Testing the parameters:")
        print(combination)
        #Suppress stdout from the Gaussian modeling function
        text_trap = io.StringIO()
        sys.stdout = text_trap

        foreground_second_part, detections = Gaussian_modelling(roi_path, video_path, alpha=combination['alpha'], rho=1,
                                                                video_length=video_length,
                                                                video_split_ratio=video_split_ratio)
        mAP = utils.calculate_mAP(gt_filtered, detections, IoU_threshold=0.5, have_confidence=False)

        # now restore stdout function
        sys.stdout = sys.__stdout__

        if mAP > bestScore:
            bestScore = mAP
            bestIteration = i

    print("The best combination of parameters, with a mAP of " + str(bestScore) + " was:")
    print(randomParameters[bestIteration])
