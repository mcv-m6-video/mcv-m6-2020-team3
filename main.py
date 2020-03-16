import os
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
import mrcnn.model as modellib

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


video_length = 200
#video_length = 2141
# video_length = 100
video_split_ratio = 0.25
# video_path = "./Datasets/AICity_data/train/S03/c010/vdo.avi"
video_path = "./Datasets/AICity/frames/"
groundtruth_xml_path = 'Datasets/AICity/aicity_annotations.xml'
# groundtruth_path = "../datasets/AICity_data/train/S03/c010/gt/gt.txt"
roi_path = 'Datasets/AICity_data/train/S03/c010/roi.jpg'
#roi = cv2.cvtColor(cv2.imread(roi_path))
noisy_gt_boxes = []

ver = True
print("Reading annotations...")
groundTruth = utw3.read_annotations(groundtruth_xml_path, video_length)
gt_filtered = [x for x in groundTruth if x['frame'] > int(video_length * video_split_ratio)]


video_first_part, video_second_part, divide_frame = \
            read_video_and_divide(video_path, video_length=video_length, video_split_ratio=video_split_ratio)
#for i in tqdm(range(video_second_part.shape[0])):
for i in tqdm(range(video_first_part.shape[0])):
    image = video_second_part[i, :, :]

#image = skimage.io.imread("C:/Users/gaby1/Pictures/source_1.jpg")

    # Run detection
    results = model.detect([image], verbose=1)
    print ('restuts:                            ')
    print(results)

    # Visualize results
    r = results[0]
    print ('results[0]')
    print (r)
    print ("r['rois']:  ")
    print (r['rois'])
    print ("r['masks']:  ")
    print (r['masks'])
    print ("r['class_ids']:")
    print (r['class_ids'])
    print ("class_names: ")
    print (class_names)
    print("r['scores']:   ")
    print(r['scores'])
    if ver:
        visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
                                class_names, r['scores'])
    # frame_id = i
    # mask = foreground_second_part[i, :, :]
    # label_image = measure.label(mask)
    # regions = measure.regionprops(label_image)
    # mask_color = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
    # for region in regions:
    #     bbox = region.bbox
    #
    #     # watch the mask
    #     box_h = bbox[2] - bbox[0]
    #     box_w = bbox[3] - bbox[1]
    #     startPoint = (int(bbox[1]), int(bbox[0]))
    #     endPoint = (int(bbox[1] + box_w), int(bbox[0] + box_h))
    #     color = (255, 0, 0)
    #     mask_color = cv2.rectangle(mask_color, startPoint, endPoint, color, 5)
    #
    #     detection = {}
    #     if filter_region(bbox, min_h, max_h, min_w, max_w, min_ratio, max_ratio):
    #         box_h = bbox[2] - bbox[0]
    #         box_w = bbox[3] - bbox[1]
    #         detection['frame'] = frame_id
    #         detection['left'] = bbox[1]
    #         detection['top'] = bbox[0]
    #         detection['width'] = box_w
    #         detection['height'] = box_h
    #         detections.append(detection)
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