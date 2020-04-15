
from utils_io import load_from_dataset
from features import compute_mr_histogram
import cv2
from metric_learn import NCA
import numpy as np
from tqdm import tqdm

import pickle
import random


def train_metric():
    datasetStructure = {
        'S01': ['c001', 'c002', 'c003', 'c004', 'c005'],
        'S03': ['c010', 'c011', 'c012', 'c013', 'c014', 'c015'],
        'S04': ['c016', 'c018', 'c020', 'c022', 'c024', 'c026', 'c028', 'c030', 'c032', 'c034', 'c036', 'c038', 'c040','c017', 'c019', 'c021', 'c023', 'c025', 'c027', 'c029', 'c031', 'c033', 'c035', 'c037', 'c039']
    }
    training_dataset = ['S01', 'S04']
    testing_dataset = ['S03']

    dataset_dicts = load_from_dataset(datasetStructure, training_dataset)


    # read the training data from the dataset.
    load_pkl = True
    if load_pkl:
        with open("features_labels.pkl", 'rb') as f:
            features, labels = pickle.load(f)
            f.close()
    else:
        features = []
        labels = []
        for frame_one in tqdm(dataset_dicts):
            image = cv2.imread(frame_one['file_name'])
            for obj in frame_one['objs']:
                box = obj["box"]
                cropped = image[box[1]:box[1]+box[3], box[0]:box[0]+box[2], :]
                # cv2.namedWindow("Image")
                # cv2.imshow("Image", cropped)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
                feature = compute_mr_histogram(cropped, splits=(3, 3), bins=32, mask=None, sqrt=False, concat=True)
                ID = obj["ID"]
                features.append(feature)
                labels.append(ID)

        with open("features_labels.pkl", 'wb') as f:
            pickle.dump([features, labels], f)
            f.close()

    # filter some detections to decrease the computing time and memory.
    # every label keeps only number_per_label detections.
    number_per_label = 10
    new_features = []
    new_labels = []
    for target_index in tqdm(range(min(labels), max(labels)+1)):
        shoot_index = [index for index, label in enumerate(labels) if label == target_index]
        if len(shoot_index) < number_per_label:
            few_index = shoot_index
        else:
            few_index = random.sample(shoot_index, number_per_label)
        selected_features = [features[x] for x in few_index]
        selected_labels = [labels[x] for x in few_index]

        new_features.extend(selected_features)
        new_labels.extend(selected_labels)


    X = np.array(new_features)
    Y = np.array(new_labels)
    nca = NCA(init='pca', n_components=400, max_iter=1000, verbose=True)
    nca.fit(X, Y)

    return nca

pairs = np.zeros((1,2,864))
pairs[0,0,:] = features[0]
pairs[0,1,:] = features[1]
nca.score_pairs(pairs)

pass