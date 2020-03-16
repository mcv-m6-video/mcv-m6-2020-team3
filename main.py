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
IMAGE_DIR = os.path.join(ROOT_DIR, "images")

# Load a random image from the images folder
file_names = next(os.walk(IMAGE_DIR))[2]
image = skimage.io.imread(os.path.join(IMAGE_DIR, random.choice(file_names)))
print (os.path.join(IMAGE_DIR, random.choice(file_names)))

# Run detection
results = model.detect([image], verbose=1)

# Visualize results
r = results[0]
visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
                            class_names, r['scores'])

'''


'''Exercise 1
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
noisy_gt_boxes = []

ver = True
print("Reading annotations...")
groundTruth = utw3.read_annotations(groundtruth_xml_path, video_length)
gt_filtered = [x for x in groundTruth if x['frame'] > int(video_length * video_split_ratio)]

video_first_part, video_second_part, divide_frame = \
    read_video_and_divide(video_path, video_length=video_length, video_split_ratio=video_split_ratio)
# for i in tqdm(range(video_second_part.shape[0])):
for i in tqdm(range(video_first_part.shape[0])):
    image = video_second_part[i, :, :]

    # image = skimage.io.imread("C:/Users/gaby1/Pictures/source_1.jpg")

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

    '''Stage 1: Region Proposal Network
    The Region Proposal Network (RPN) runs a lightweight binary classifier on a lot of boxes (anchors) over the image and returns object/no-object scores. Anchors with high objectness score (positive anchors) are passed to the stage two to be classified.
    
    Often, even positive anchors don't cover objects fully. So the RPN also regresses a refinement (a delta in location and size) to be applied to the anchors to shift it and resize it a bit to the correct boundaries of the object.
    
    1.a RPN Predictions
    Here we run the RPN graph and display its predictions.
    '''
    # Run RPN sub-graph
    pillar = model.keras_model.get_layer("ROI").output  # node to start searching from
    h, w = image.shape[:2]
    # TF 1.4 and 1.9 introduce new versions of NMS. Search for all names to support TF 1.3~1.10
    nms_node = model.ancestor(pillar, "ROI/rpn_non_max_suppression:0")
    # if nms_node is None:
    #     nms_node = model.ancestor(pillar, "ROI/rpn_non_max_suppression/NonMaxSuppressionV2:0")
    if nms_node is None:  # TF 1.9-1.10
        nms_node = model.ancestor(pillar, "ROI/rpn_non_max_suppression/NonMaxSuppressionV3:0")

    rpn = model.run_graph([image], [
        ("rpn_class", model.keras_model.get_layer("rpn_class").output),
        ("pre_nms_anchors", model.ancestor(pillar, "ROI/pre_nms_anchors:0")),
        ("refined_anchors", model.ancestor(pillar, "ROI/refined_anchors:0")),
        ("refined_anchors_clipped", model.ancestor(pillar, "ROI/refined_anchors_clipped:0")),
        ("post_nms_anchor_ix", nms_node),
        ("proposals", model.keras_model.get_layer("ROI").output),
    ])
    # Show top anchors with refinement. Then with clipping to image boundaries
    limit = 50
    ax = get_ax(1, 2)
    shape = (image.shape[0], image.shape[1])
    pre_nms_anchors = utils.denorm_boxes(rpn["pre_nms_anchors"][0], shape)
    refined_anchors = utils.denorm_boxes(rpn["refined_anchors"][0], shape)
    refined_anchors_clipped = utils.denorm_boxes(rpn["refined_anchors_clipped"][0], shape)
    visualize.draw_boxes(image, boxes=pre_nms_anchors[:limit],
                         refined_boxes=refined_anchors[:limit], ax=ax[0])
    visualize.draw_boxes(image, refined_boxes=refined_anchors_clipped[:limit], ax=ax[1])

    # Show refined anchors after non-max suppression
    limit = 50
    ixs = rpn["post_nms_anchor_ix"][:limit]
    visualize.draw_boxes(image, refined_boxes=refined_anchors_clipped[ixs], ax=get_ax())

    # Show final proposals
    # These are the same as the previous step (refined anchors
    # after NMS) but with coordinates normalized to [0, 1] range.
    limit = 50
    # Convert back to image coordinates for display
    h, w = image.shape[:2]
    proposals = rpn['proposals'][0, :limit] * np.array([h, w, h, w])
    visualize.draw_boxes(image, refined_boxes=proposals, ax=get_ax())

    '''Stage 2: Proposal Classification
    This stage takes the region proposals from the RPN and classifies them.
    
    2.a Proposal Classification
    Run the classifier heads on proposals to generate class propbabilities and bounding box regressions
    '''
    # Get input and output to classifier and mask heads.
    mrcnn = model.run_graph([image], [
        ("proposals", model.keras_model.get_layer("ROI").output),
        ("probs", model.keras_model.get_layer("mrcnn_class").output),
        ("deltas", model.keras_model.get_layer("mrcnn_bbox").output),
        ("masks", model.keras_model.get_layer("mrcnn_mask").output),
        ("detections", model.keras_model.get_layer("mrcnn_detection").output),
    ])

    # Get detection class IDs. Trim zero padding.
    det_class_ids = mrcnn['detections'][0, :, 4].astype(np.int32)
    det_count = np.where(det_class_ids == 0)[0][0]
    det_class_ids = det_class_ids[:det_count]
    detections = mrcnn['detections'][0, :det_count]

    print("{} detections: {}".format(
        det_count, np.array(class_names)[det_class_ids]))

    captions = ["{} {:.3f}".format(class_names[int(c)], s) if c > 0 else ""
                for c, s in zip(detections[:, 4], detections[:, 5])]
    visualize.draw_boxes(
        image,
        refined_boxes=utils.denorm_boxes(detections[:, :4], image.shape[:2]),
        visibilities=[2] * len(detections),
        captions=captions, title="Detections",
        ax=get_ax())

    '''2.b Step by Step Detection
    Here we dive deeper into the process of processing the detections.
    '''

    # Proposals are in normalized coordinates. Scale them
    # to image coordinates.
    h, w = image.shape[:2]
    proposals = np.around(mrcnn["proposals"][0] * np.array([h, w, h, w])).astype(np.int32)

    # Class ID, score, and mask per proposal
    roi_class_ids = np.argmax(mrcnn["probs"][0], axis=1)
    roi_scores = mrcnn["probs"][0, np.arange(roi_class_ids.shape[0]), roi_class_ids]
    roi_class_names = np.array(class_names)[roi_class_ids]
    roi_positive_ixs = np.where(roi_class_ids > 0)[0]

    # How many ROIs vs empty rows?
    print("{} Valid proposals out of {}".format(np.sum(np.any(proposals, axis=1)), proposals.shape[0]))
    print("{} Positive ROIs".format(len(roi_positive_ixs)))

    # Class counts
    print(list(zip(*np.unique(roi_class_names, return_counts=True))))

    # Display a random sample of proposals.
    # Proposals classified as background are dotted, and
    # the rest show their class and confidence score.
    h, w = image.shape[:2]
    limit = 200
    ixs = np.random.randint(0, proposals.shape[0], limit)
    captions = ["{} {:.3f}".format(class_names[c], s) if c > 0 else ""
                for c, s in zip(roi_class_ids[ixs], roi_scores[ixs])]
    visualize.draw_boxes(image, boxes=proposals[ixs],
                         visibilities=np.where(roi_class_ids[ixs] > 0, 2, 1),
                         captions=captions, title="ROIs Before Refinement",
                         ax=get_ax())

    '''Apply Bounding Box Refinement
    '''
    # Class-specific bounding box shifts.
    roi_bbox_specific = mrcnn["deltas"][0, np.arange(proposals.shape[0]), roi_class_ids]
    log("roi_bbox_specific", roi_bbox_specific)

    # Apply bounding box transformations
    # Shape: [N, (y1, x1, y2, x2)]
    refined_proposals = utils.apply_box_deltas(
        proposals, roi_bbox_specific * config.BBOX_STD_DEV).astype(np.int32)
    log("refined_proposals", refined_proposals)

    # Show positive proposals
    # ids = np.arange(roi_boxes.shape[0])  # Display all
    limit = 5
    ids = np.random.randint(0, len(roi_positive_ixs), limit)  # Display random sample
    captions = ["{} {:.3f}".format(class_names[c], s) if c > 0 else ""
                for c, s in zip(roi_class_ids[roi_positive_ixs][ids], roi_scores[roi_positive_ixs][ids])]
    visualize.draw_boxes(image, boxes=proposals[roi_positive_ixs][ids],
                         refined_boxes=refined_proposals[roi_positive_ixs][ids],
                         visibilities=np.where(roi_class_ids[roi_positive_ixs][ids] > 0, 1, 0),
                         captions=captions, title="ROIs After Refinement",
                         ax=get_ax())

    '''Filter Low Confidence Detections'''
    # Remove boxes classified as background
    keep = np.where(roi_class_ids > 0)[0]
    print("Keep {} detections:\n{}".format(keep.shape[0], keep))
    # Remove low confidence detections
    keep = np.intersect1d(keep, np.where(roi_scores >= config.DETECTION_MIN_CONFIDENCE)[0])
    print("Remove boxes below {} confidence. Keep {}:\n{}".format(
        config.DETECTION_MIN_CONFIDENCE, keep.shape[0], keep))

    '''Per-Class Non-Max Suppression'''

    # Apply per-class non-max suppression
    pre_nms_boxes = refined_proposals[keep]
    pre_nms_scores = roi_scores[keep]
    pre_nms_class_ids = roi_class_ids[keep]

    nms_keep = []
    for class_id in np.unique(pre_nms_class_ids):
        # Pick detections of this class
        ixs = np.where(pre_nms_class_ids == class_id)[0]
        # Apply NMS
        class_keep = utils.non_max_suppression(pre_nms_boxes[ixs],
                                               pre_nms_scores[ixs],
                                               config.DETECTION_NMS_THRESHOLD)
        # Map indicies
        class_keep = keep[ixs[class_keep]]
        nms_keep = np.union1d(nms_keep, class_keep)
        print("{:22}: {} -> {}".format(class_names[class_id][:20],
                                       keep[ixs], class_keep))

    keep = np.intersect1d(keep, nms_keep).astype(np.int32)
    print("\nKept after per-class NMS: {}\n{}".format(keep.shape[0], keep))
    # Show final detections
    ixs = np.arange(len(keep))  # Display all
    # ixs = np.random.randint(0, len(keep), 10)  # Display random sample
    captions = ["{} {:.3f}".format(class_names[c], s) if c > 0 else ""
                for c, s in zip(roi_class_ids[keep][ixs], roi_scores[keep][ixs])]
    visualize.draw_boxes(
        image, boxes=proposals[keep][ixs],
        refined_boxes=refined_proposals[keep][ixs],
        visibilities=np.where(roi_class_ids[keep][ixs] > 0, 1, 0),
        captions=captions, title="Detections after NMS",
        ax=get_ax())

    '''Stage 3: Generating Masks
    This stage takes the detections (refined bounding boxes and class IDs) from the previous layer and runs the mask head to generate segmentation masks for every instance.
    
    3.a Predicted Masks
    '''

    # Get predictions of mask head
    mrcnn = model.run_graph([image], [
        ("detections", model.keras_model.get_layer("mrcnn_detection").output),
        ("masks", model.keras_model.get_layer("mrcnn_mask").output),
    ])

    # Get detection class IDs. Trim zero padding.
    det_class_ids = mrcnn['detections'][0, :, 4].astype(np.int32)
    det_count = np.where(det_class_ids == 0)[0][0]
    det_class_ids = det_class_ids[:det_count]

    print("{} detections: {}".format(
        det_count, np.array(class_names)[det_class_ids]))

    # Masks
    det_boxes = utils.denorm_boxes(mrcnn["detections"][0, :, :4], image.shape[:2])
    det_mask_specific = np.array([mrcnn["masks"][0, i, :, :, c]
                                  for i, c in enumerate(det_class_ids)])
    det_masks = np.array([utils.unmold_mask(m, det_boxes[i], image.shape)
                          for i, m in enumerate(det_mask_specific)])
    log("det_mask_specific", det_mask_specific)
    log("det_masks", det_masks)

    display_images(det_mask_specific[:6] * 255, cmap="Blues", interpolation="none")
    display_images(det_masks[:4] * 255, cmap="Blues", interpolation="none")

    '''Visualize Activations
    In some cases it helps to look at the output from different layers and visualize them to catch issues and odd patterns.'''

    # Get activations of a few sample layers
    activations = model.run_graph([image], [
        ("input_image", tf.identity(model.keras_model.get_layer("input_image").output)),
        ("res4w_out", model.keras_model.get_layer("res4w_out").output),  # for resnet100
        ("rpn_bbox", model.keras_model.get_layer("rpn_bbox").output),
        ("roi", model.keras_model.get_layer("ROI").output),
    ])

    # Input image (normalized)
    _ = plt.imshow(modellib.unmold_image(activations["input_image"][0], config))

    # Backbone feature map
    display_images(np.transpose(activations["res4w_out"][0, :, :, :4], [2, 0, 1]))

    # Histograms of RPN bounding box deltas
    plt.figure(figsize=(12, 3))
    plt.subplot(1, 4, 1)
    plt.title("dy")
    _ = plt.hist(activations["rpn_bbox"][0, :, 0], 50)
    plt.subplot(1, 4, 2)
    plt.title("dx")
    _ = plt.hist(activations["rpn_bbox"][0, :, 1], 50)
    plt.subplot(1, 4, 3)
    plt.title("dw")
    _ = plt.hist(activations["rpn_bbox"][0, :, 2], 50)
    plt.subplot(1, 4, 4)
    plt.title("dh")
    _ = plt.hist(activations["rpn_bbox"][0, :, 3], 50)

    # Histograms of RPN bounding box deltas
    plt.figure(figsize=(12, 3))
    plt.subplot(1, 4, 1)
    plt.title("dy")
    _ = plt.hist(activations["rpn_bbox"][0, :, 0], 50)
    plt.subplot(1, 4, 2)
    plt.title("dx")
    _ = plt.hist(activations["rpn_bbox"][0, :, 1], 50)
    plt.subplot(1, 4, 3)
    plt.title("dw")
    _ = plt.hist(activations["rpn_bbox"][0, :, 2], 50)
    plt.subplot(1, 4, 4)
    plt.title("dh")
    _ = plt.hist(activations["rpn_bbox"][0, :, 3], 50)
