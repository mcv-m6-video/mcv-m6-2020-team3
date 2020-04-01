import os
import numpy as np
import sys
import random
import tensorflow as tf
import matplotlib.pyplot as plt
import skimage
import cv2
import pickle
import utils as ut
from utils_Gaussian import read_video_and_divide, calculate_mean_std_first_part_video, calculate_mask, find_detections
from tqdm import tqdm
from AICityConfig import AICityConfig
from AICityDataset import AICityDataset
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

def get_ax(rows=1, cols=1, size=16):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.

    Adjust the size attribute to control how big to render images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size * cols, size * rows))
    return ax

#%matplotlib inline
class InferenceConfig(AICityConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

# # Path to Shapes trained weights
SHAPES_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_shapes.h5")

# video_length = 200
video_length = 2141
video_split_ratio = 0.25
video_path = "./Datasets/AICity/frames"
groundtruth_xml_path = 'Datasets/AICity/aicity_annotations.xml'
showImages = False

config = AICityConfig()
config.display()
# Training dataset
dataset_train = AICityDataset()
dataset_train.get_Images(framePath=video_path, length=video_length, isTrain=True, trainSplit=0.25, height=config.IMAGE_SHAPE[0], width=config.IMAGE_SHAPE[1])
dataset_train.prepare()
dataset_val = AICityDataset()
dataset_val.get_Images(framePath=video_path, length=video_length, isTrain=False, trainSplit=0.25, height=config.IMAGE_SHAPE[0], width=config.IMAGE_SHAPE[1])
dataset_val.prepare()

# Create model in training mode
model = modellib.MaskRCNN(mode="training", config=config,
                          model_dir=MODEL_DIR)

# Which weights to start with?
init_with = "coco"  # imagenet, coco, or last

if init_with == "imagenet":
    model.load_weights(model.get_imagenet_weights(), by_name=True)
elif init_with == "coco":
    # Load weights trained on MS COCO, but skip layers that
    # are different due to the different number of classes
    # See README for instructions to download the COCO weights
    model.load_weights(COCO_MODEL_PATH, by_name=True,
                       exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",
                                "mrcnn_bbox", "mrcnn_mask"])
elif init_with == "last":
    # Load the last model you trained and continue training
    model.load_weights(model.find_last(), by_name=True)

# Train the head branches
# Passing layers="heads" freezes all layers except the head
# layers. You can also pass a regular expression to select
# which layers to train by name pattern.
model.train(dataset_train, dataset_val,
            learning_rate=config.LEARNING_RATE,
            epochs=10,
            layers='heads')

# Fine tune all layers
# Passing layers="all" trains all layers. You can also
# pass a regular expression to select which layers to
# train by name pattern.
#model.train(dataset_train, dataset_val, learning_rate=config.LEARNING_RATE / 10, epochs=2, layers="all")

inference_config = InferenceConfig()

# Recreate the model in inference mode
model = modellib.MaskRCNN(mode="inference",
                          config=inference_config,
                          model_dir=MODEL_DIR)

# Get path to saved weights
# Either set a specific path or find last trained weights
# model_path = os.path.join(ROOT_DIR, ".h5 file name here")
model_path = model.find_last()

# Load trained weights
print("Loading weights from ", model_path)
model.load_weights(model_path, by_name=True)

# Test on a random image
image_id_predict = random.choice(dataset_val.image_ids)
original_image, image_meta, gt_class_id, gt_bbox, gt_mask =\
    modellib.load_image_gt(dataset_val, inference_config,
                           image_id_predict, use_mini_mask=False)

log("original_image", original_image)
log("image_meta", image_meta)
log("gt_class_id", gt_class_id)
log("gt_bbox", gt_bbox)
log("gt_mask", gt_mask)

ut.save_instances(original_image, gt_bbox, gt_mask, gt_class_id,
                    dataset_train.class_names, figsize=(8, 8), imName='gt_' + str(image_id_predict) + '.png')


results = model.detect([original_image], verbose=1)

r = results[0]

ut.save_instances(original_image, r['rois'], r['masks'], r['class_ids'],
                        dataset_val.class_names, r['scores'], ax=get_ax(), imName='pred_' + str(image_id_predict) + '.png')

# Compute VOC-Style mAP @ IoU=0.5
# Running on 10 images. Increase for better accuracy.
#image_ids = np.random.choice(dataset_val.image_ids, 100)
APs = []
print("Evaluating fine-tuned model")
resultsPkl = []
for image_id in tqdm(dataset_val.image_ids):
    #Load image and ground truth data
    image, image_meta, gt_class_id, gt_bbox, gt_mask = \
        modellib.load_image_gt(dataset_val, inference_config,
                               image_id, use_mini_mask=False)
    molded_images = np.expand_dims(modellib.mold_image(image, inference_config), 0)
    #Run object detection
    results = model.detect([image], verbose=0)
    resultsPkl.append(results)
    r = results[0]
    #Compute AP
    AP, precisions, recalls, overlaps = \
        utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
                         r["rois"], r["class_ids"], r["scores"], r['masks'])
    APs.append(AP)

print("Saving results to pickle")

detections = []
for i, result in tqdm(enumerate(resultsPkl), total=len(resultsPkl)):
    r = result[0]
    class_id = r['class_ids']
    # print (class_id)
    boxes = r['rois']
    confidence = r['scores']
    for j in range(class_id.shape[0]):
        if class_id[j] == 1:
            bbox = boxes[j, :]
            detection = {}
            box_h = bbox[2] - bbox[0]
            box_w = bbox[3] - bbox[1]
            detection['frame'] = dataset_val.image_info[i]['id']
            detection['left'] = bbox[1]
            detection['top'] = bbox[0]
            detection['width'] = box_w
            detection['height'] = box_h
            detection['confidence'] = confidence[j]
            detections.append(detection)

with open('detections.pkl', "wb") as f:
    pickle.dump(detections, f)

print("mAP: ", np.mean(APs))
print("Images with GT and predictions in gt_" + str(image_id_predict) + ".png and pred_" + str(image_id_predict) + ".png")
