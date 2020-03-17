import os
import numpy as np
import sys
import random
import tensorflow as tf
import matplotlib.pyplot as plt
import skimage
import cv2
import utils_w3 as utw3
from utils_Gaussian import read_video_and_divide, calculate_mean_std_first_part_video, calculate_mask, find_detections
from tqdm import tqdm
# Root directory of the project
ROOT_DIR = os.path.abspath("./Mask_RCNN")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
from mrcnn import visualize
from mrcnn.config import Config
from mrcnn.visualize import display_images
import mrcnn.model as modellib
from mrcnn.model import log


#%matplotlib inline

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

# # Path to Shapes trained weights
SHAPES_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_shapes.h5")
sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))
print (ROOT_DIR)
import coco
config = coco.CocoConfig()
# Override the training configurations with a few
# changes for inferencing.
class InferenceConfig(config.__class__):
    # Run detection on one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
config.display()
DEVICE = "/gpu:0"  # /cpu:0 or /gpu:0
TEST_MODE = "inference"


def get_ax(rows=1, cols=1, size=16):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.

    Adjust the size attribute to control how big to render images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size * cols, size * rows))
    return ax
# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']

with tf.device(DEVICE):
    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config = config)

# Set weights file path
if config.NAME == "shapes":
    weights_path = SHAPES_MODEL_PATH
elif config.NAME == "coco":
    weights_path = COCO_MODEL_PATH
# Or, uncomment to load the last model you trained
# weights_path = model.find_last()

# Load weights
print("Loading weights ", weights_path)
model.load_weights(weights_path, by_name=True)


'''
Test the network with different images taken from our dataset. Take into account that the network can only predict classes from COCO.
'''
video_length = 200
# video_length = 2141
# video_length = 100
video_split_ratio = 0.25
# video_path = "./Datasets/AICity_data/train/S03/c010/vdo.avi"
video_path = "./Datasets/AICity/frames/"
groundtruth_xml_path = 'Datasets/AICity/aicity_annotations.xml'
# groundtruth_path = "../datasets/AICity_data/train/S03/c010/gt/gt.txt"
roi_path = 'Datasets/AICity_data/train/S03/c010/roi.jpg'
# roi = cv2.cvtColor(cv2.imread(roi_path))
ver = True
print("Reading annotations...")
groundTruth = utw3.read_annotations(groundtruth_xml_path, video_length)
gt_filtered = [x for x in groundTruth if x['frame'] > int(video_length * video_split_ratio)]

video_first_part, video_second_part, divide_frame = \
    read_video_and_divide(video_path, video_length=video_length, video_split_ratio=video_split_ratio)
# for i in tqdm(range(video_second_part.shape[0])):
for i in tqdm(range(video_first_part.shape[0])):
    image = video_second_part[i, :, :]
    # Run detection
    results = model.detect([image], verbose=1)
    print('restuts:                            ')
    print(results)

    # Visualize results
    r = results[0]
    print('results[0]')
    print(r)
    print("r['rois']:  ")
    print(r['rois'])
    print("r['masks']:  ")
    print(r['masks'])
    print("r['class_ids']:")
    print(r['class_ids'])
    print("class_names: ")
    print(class_names)
    print("r['scores']:   ")
    print(r['scores'])
    if ver:
        visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
                                    class_names, r['scores'])
