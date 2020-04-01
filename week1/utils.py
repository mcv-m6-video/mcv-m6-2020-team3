import os
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as pat
from xml.dom import minidom
import cv2
import glob
from tqdm import tqdm
import xml.etree.ElementTree as ET
from copy import deepcopy
from collections import defaultdict
from numpy import random
from skimage.measure import label, regionprops
# from detection import Detection

def getDetections(detectionFilePath):
    with open(detectionFilePath, 'r') as f:
        detections = [getDictFromDetection(line) for line in f]
    return detections
def getDictFromDetection(detectionStr):
    detectionList = detectionStr.split(",")
    detectionDict = {}
    detectionDict['frame'] = int(float(detectionList[0]))
    detectionDict['left'] = int(float(detectionList[2]))
    detectionDict['top'] = int(float(detectionList[3]))
    detectionDict['width'] = int(float(detectionList[4]))
    detectionDict['height'] = int(float(detectionList[5]))
    detectionDict['confidence'] = float(detectionList[6])
    return detectionDict
def bb_iou(bboxA, bboxB):
    # This implements a function to compute the intersection over union of two bounding boxes, also known as the Jaccard Index.
    # I've adapted this code from the M1 project code we implemented. The Format of the bboxes is [tlx, tly, brx, bry, ...], where tl and br
    # indicate top-left and bottom-right corners of the bbox respectively.

    # computing coordinates of the intersection rectangle

    xA = max(bboxA[1], bboxB[1])
    yA = max(bboxA[0], bboxB[0])
    xB = min(bboxA[3], bboxB[3])
    yB = min(bboxA[2], bboxB[2])

    # compute the area of intersection rectangle
    
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both predicted and ground-truth bboxes
    
    bboxAArea = (bboxA[2] - bboxA[0] + 1) * (bboxA[3] - bboxA[1] + 1)
    bboxBArea = (bboxB[2] - bboxB[0] + 1) * (bboxB[3] - bboxB[1] + 1)

    try:
        iou = interArea / float(bboxAArea + bboxBArea - interArea)
    except ZeroDivisionError:
        iou = 0.0

    # returns the intersection over union value
    return iou
def get_noisy_bboxes(discard_probability_bbox=0.1, noise_range=20):
    """
    The function returns a dictionary with the bounding boxes for each frame where the frame number is the key. It also returns a dictionary
    that contains noisy annotations to the ground truth data. 
    """
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    with open(ROOT_DIR+'/../Datasets/AICity/train/S03/c010/gt/gt.txt') as f:
        lines = f.readlines()
        bboxes = dict() # stores the ground truth bboxes for each frame in the dict
        bboxes_noisy = dict() # stores the noisy annotations to the ground truth data
        num_of_instances = 0 
        for line in lines:
            num_of_instances += 1
            line = (line.split(','))
            if line[0] in bboxes.keys():
                content = [int(elem) for elem in line[1:6]]
                content.append(random())
                bboxes[line[0]].append(content)
            else:
                content = [int(elem) for elem in line[1:6]]
                content.append(random())
                bboxes[line[0]] = [content]
            if random() > discard_probability_bbox:
                if line[0] in bboxes_noisy.keys():
                    content = [int(elem) + randrange(-noise_range, noise_range) for elem in line[1:6]]
                    content.append(random())
                    bboxes_noisy[line[0]].append(content)
                else:
                    content = [int(elem) + randrange(-noise_range, noise_range) for elem in line[1:6]]
                    content.append(random())
                    bboxes_noisy[line[0]] = [content]

    return bboxes, bboxes_noisy, num_of_instances
def read_xml_annotations(annotations_path):
    '''The function reads the xml files and returns the corresponding bboxes for every frame.'''
    files = os.listdir(annotations_path)

    bboxes = dict()
    for file_ in files:
        xmldoc = minidom.parse(annotations_path + file_)
        bboxes_list = xmldoc.getElementsByTagName('box')
        for element in bboxes_list:
            frame = element.getAttribute('frame')
            tlx = int(float(element.getAttribute('tlx')))
            tly = int(float(element.getAttribute('tly')))
            brx = int(float(element.getAttribute('brx')))
            bry = int(float(element.getAttribute('bry')))
            width = ybr - ytl
            height = xbr - xtl

            if frame in bboxes.keys():
                bboxes[frame].append([-1, xtl, ytl, height, width, random()])
            else:
                bboxes[frame] = [[-1, xtl, ytl, height, width, random()]]
    return bboxes
def get_single_frame_results(gt_boxes, pred_boxes, iou_thr):
    """Calculates number of true_pos, false_pos, false_neg from single batch of boxes.
    Args:
        gt_boxes (list of list of floats): list of locations of ground truth
            objects as [xmin, ymin, xmax, ymax]
        pred_boxes (dict): dict of dicts of 'boxes' (formatted like `gt_boxes`)
            and 'scores'
        iou_thr (float): value of IoU to consider as threshold for a
            true prediction.
    Returns:
        dict: true positives (int), false positives (int), false negatives (int)
    """

    all_pred_indices = range(len(pred_boxes))
    all_gt_indices = range(len(gt_boxes))
    if len(all_pred_indices) == 0:
        tp = 0
        fp = 0
        fn = len(gt_boxes)
        return {'true_pos': tp, 'false_pos': fp, 'false_neg': fn}
    if len(all_gt_indices) == 0:
        tp = 0
        fp = len(pred_boxes)
        fn = 0
        return {'true_pos': tp, 'false_pos': fp, 'false_neg': fn}

    gt_idx_thr = []
    pred_idx_thr = []
    ious = []
    for ipb, pred_box in enumerate(pred_boxes):
        for igb, gt_box in enumerate(gt_boxes):
            iou = bb_iou(pred_box, gt_box)
            if iou > iou_thr:
                gt_idx_thr.append(igb)
                pred_idx_thr.append(ipb)
                ious.append(iou)

    args_desc = np.argsort(ious)[::-1]
    if len(args_desc) == 0:
        # No matches
        tp = 0
        fp = len(pred_boxes)
        fn = len(gt_boxes)
    else:
        gt_match_idx = []
        pred_match_idx = []
        for idx in args_desc:
            gt_idx = gt_idx_thr[idx]
            pr_idx = pred_idx_thr[idx]
            # If the boxes are unmatched, add them to matches
            if (gt_idx not in gt_match_idx) and (pr_idx not in pred_match_idx):
                gt_match_idx.append(gt_idx)
                pred_match_idx.append(pr_idx)
        tp = len(gt_match_idx)
        fp = len(pred_boxes) - len(pred_match_idx)
        fn = len(gt_boxes) - len(gt_match_idx)

    return {'true_pos': tp, 'false_pos': fp, 'false_neg': fn}
def calc_precision_recall(img_results):
    """Calculates precision and recall from the set of frames in video
    Args:
        img_results (dict): dictionary formatted like:
            {
                'img_id1': {'true_pos': int, 'false_pos': int, 'false_neg': int},
                'img_id2': ...
                ...
            }
    Returns:
        tuple: of floats of (precision, recall)
    """
    true_pos = 0; false_pos = 0; false_neg = 0
    for _, res in img_results.items():
        true_pos += res['true_pos']
        false_pos += res['false_pos']
        false_neg += res['false_neg']

    try:
        precision = true_pos/(true_pos + false_pos)
    except ZeroDivisionError:
        precision = 0.0
    try:
        recall = true_pos/(true_pos + false_neg)
    except ZeroDivisionError:
        recall = 0.0

    return (precision, recall)
def get_model_scores_map(pred_boxes):
    """Creates a dictionary of from model_scores to frame ids.
    Args:
        pred_boxes (dict): dict of dicts of 'boxes' and 'scores'
    Returns:
        dict: keys are model_scores and values are image ids (usually filenames)
    """
    model_scores_map = {}
    for img_id, val in pred_boxes.items():
        for score in val['scores']:
            if score not in model_scores_map.keys():
                model_scores_map[score] = [img_id]
            else:
                model_scores_map[score].append(img_id)
    return model_scores_map
def get_avg_precision_at_iou(gt_boxes, pred_boxes, iou_thr=0.5):
    """Calculates average precision at given IoU threshold.
    Args:
        gt_boxes (list of list of floats): list of locations of ground truth
            objects as [xmin, ymin, xmax, ymax]
        pred_boxes (list of list of floats): list of locations of predicted
            objects as [xmin, ymin, xmax, ymax]
        iou_thr (float): value of IoU to consider as threshold for a
            true prediction.
    Returns:
        dict: avg precision as well as summary info about the PR curve
        Keys:
            'avg_prec' (float): average precision for this IoU threshold
            'precisions' (list of floats): precision value for the given
                model_threshold
            'recall' (list of floats): recall value for given
                model_threshold
            'models_thrs' (list of floats): model threshold value that
                precision and recall were computed for.
    """
    model_scores_map = get_model_scores_map(pred_boxes)
    sorted_model_scores = sorted(model_scores_map.keys())

    # Sort the predicted boxes in descending order (lowest scoring boxes first):
    for img_id in pred_boxes.keys():
        arg_sort = np.argsort(pred_boxes[img_id]['scores'])
        pred_boxes[img_id]['scores'] = np.array(pred_boxes[img_id]['scores'])[arg_sort].tolist()
        pred_boxes[img_id]['boxes'] = np.array(pred_boxes[img_id]['boxes'])[arg_sort].tolist()

    pred_boxes_pruned = deepcopy(pred_boxes)

    precisions = []
    recalls = []
    model_thrs = []
    img_results = {}
    # Loop over model score thresholds and calculate precision, recall
    for ithr, model_score_thr in enumerate(sorted_model_scores[:-1]):
        # On first iteration, define img_results for the first time:
        img_ids = gt_boxes.keys() if ithr == 0 else model_scores_map[model_score_thr]
        for img_id in img_ids:
            gt_boxes_img = gt_boxes[img_id]
            box_scores = pred_boxes_pruned[img_id]['scores']
            start_idx = 0
            for score in box_scores:
                if score <= model_score_thr:
                    pred_boxes_pruned[img_id]
                    start_idx += 1
                else:
                    break

            # Remove boxes, scores of lower than threshold scores:
            pred_boxes_pruned[img_id]['scores'] = pred_boxes_pruned[img_id]['scores'][start_idx:]
            pred_boxes_pruned[img_id]['boxes'] = pred_boxes_pruned[img_id]['boxes'][start_idx:]

            # Recalculate image results for this image
            img_results[img_id] = get_single_frame_results(
                gt_boxes_img, pred_boxes_pruned[img_id]['boxes'], iou_thr)

        prec, rec = calc_precision_recall(img_results)
        precisions.append(prec)
        recalls.append(rec)
        model_thrs.append(model_score_thr)

    precisions = np.array(precisions)
    recalls = np.array(recalls)
    prec_at_rec = []
    for recall_level in np.linspace(0.0, 1.0, 11):
        try:
            args = np.argwhere(recalls >= recall_level).flatten()
            prec = max(precisions[args])
        except ValueError:
            prec = 0.0
        prec_at_rec.append(prec)
    avg_prec = np.mean(prec_at_rec)

    return {
        'avg_prec': avg_prec,
        'precisions': precisions,
        'recalls': recalls,
        'model_thrs': model_thrs}
def plot_pr_curve(precisions, recalls, category='Cars', label=None, color=None, ax=None):
    """Simple plotting helper function"""

    if ax is None:
        plt.figure(figsize=(10,8))
        ax = plt.gca()

    if color is None:
        color = COLORS[0]
    ax.scatter(recalls, precisions, label=label, s=20, color=color)
    ax.set_xlabel('recall')
    ax.set_ylabel('precision')
    ax.set_title('Precision-Recall curve for {}'.format(category))
    ax.set_xlim([0.0,1.3])
    ax.set_ylim([0.0,1.2])
    return ax
def sortDetectionsByKey(detectionList, key, decreasing=False):
    sortedList = sorted(detectionList, key=lambda k: k[key], reverse=decreasing)
    return sortedList
def shuffle_detectionList(detectionList):
    random.shuffle(detectionList)
    return detectionList
def getBboxFromDetection(detection):
    bbox = np.zeros(4)
    bbox[0] = detection['left']
    bbox[1] = detection['top']
    bbox[2] = detection['left'] + detection['width']
    bbox[3] = detection['left'] + detection['height']
    return bbox
def addBboxesToFrames(framesPath, detections, groundTruth, name):
    #Show GT bboxes and detections
    #Preprocess detections and GT
    for detection in detections:
        detection['isGT'] = False
    for item in groundTruth:
        item['isGT'] = True

    combinedList = detections + groundTruth

    combinedList = sortDetectionsByKey(combinedList, 'frame')

    frameFiles = glob.glob(framesPath + '/*.jpg')
    size = (1920, 1080)
    fps = 10
    out = cv2.VideoWriter(name + '.avi',cv2.VideoWriter_fourcc(*'DIVX'), fps, size)

    prevFrame = 0
    frameMat = cv2.imread(frameFiles[0])
    for item in tqdm(combinedList):
        frame = item['frame'] - 1
        if frame != prevFrame:
            out.write(frameMat)
            frameMat = cv2.imread(frameFiles[frame])
        startPoint = (int(item['left']), int(item['top']))
        endPoint = (int(startPoint[0] + item['width']), int(startPoint[1] + item['height']))
        color = (255, 0, 0) if item['isGT'] is True else (0, 0, 255)
        frameMat = cv2.rectangle(frameMat, startPoint, endPoint, color, 2)
        prevFrame = frame
    out.release()
def getDetectionsPerFrame(detections):
    detectionDict = {}
    for detection in detections:
        if detection['frame'] not in detectionDict.keys():
            detectionDict[detection['frame']] = [detection]
        else:
            detectionDict[detection['frame']].append(detection)
    return detectionDict
def read_annotations(annotation_path, video_len):
    """
    Arguments:
    capture: frames from video, opened as cv2.VideoCapture
    root: parsed xml annotations as ET.parse(annotation_path).getroot()
    """
    root = ET.parse(annotation_path).getroot()
    groundTruth = []

    for frame in tqdm(range(video_len)):
        for track in root.findall('track'):
            label = track.attrib['label']
            box = track.find("box[@frame='{0}']".format(str(frame)))
            if box is not None and label == 'car':
                detectionDict = {}
                detectionDict['frame'] = int(box.attrib['frame']) + 1
                detectionDict['left'] = int(float(box.attrib['xtl']))
                detectionDict['top'] = int(float(box.attrib['ytl']))
                detectionDict['width'] = int(float(box.attrib['xbr'])) - int(float(box.attrib['xtl']))
                detectionDict['height'] = int(float(box.attrib['ybr'])) - int(float(box.attrib['ytl']))
                groundTruth.append(detectionDict)

    return groundTruth
def add_noise_to_detections(gt_boxes_path, video_len, rescaling_factor = [0.5, 1], translation_factor = 30, prob_discard = 0.1):
    noisy_gt_boxes = []
    # rescaling_factor = [0.5, 1]
    # translation_factor = 30
    # prob_discard = 0.1
    root = ET.parse(gt_boxes_path).getroot()
    groundTruth = []

    for frame in tqdm(range(video_len)):
        for track in root.findall('track'):
            label = track.attrib['label']
            box = track.find("box[@frame='{0}']".format(str(frame)))
            if box is not None and label == 'car':
                detectionDict = {}
                detectionDict['frame'] = int(box.attrib['frame']) + 1
                detectionDict['left'] = int(float(box.attrib['xtl']))
                detectionDict['top'] = int(float(box.attrib['ytl']))
                detectionDict['width'] = int(float(box.attrib['xbr'])) - int(float(box.attrib['xtl']))
                detectionDict['height'] = int(float(box.attrib['ybr'])) - int(float(box.attrib['ytl']))
                # groundTruth.append(detectionDict)
                if np.random.uniform(0, 1) < prob_discard:
                    continue
                # tl_x, tl_y = detectionDict['left'], detectionDict['top']
                detectionDict['left'] += np.random.uniform(0, 1) * translation_factor
                detectionDict['top'] += np.random.uniform(0, 1) * translation_factor
                detectionDict['width'] = detectionDict['width'] * np.random.uniform(rescaling_factor[0], rescaling_factor[1])
                detectionDict['height'] = detectionDict['height'] * np.random.uniform(rescaling_factor[0], rescaling_factor[1])

                noisy_gt_boxes.append(detectionDict)

    return noisy_gt_boxes
def box(o):
    return [o['left'], o['top'], o['left'] + o['width'], o['top'] + o['height']]
def calculate_mAP(groundtruth_list_original, detections_list, IoU_threshold=0.5, have_confidence = True, verbose = False):

    groundtruth_list = deepcopy(groundtruth_list_original)

    # Sort detections by confidence
    if have_confidence:
        detections_list.sort(key=lambda x: x['confidence'], reverse=True)
    # Save number of groundtruth labels
    groundtruth_size = len(groundtruth_list)

    TP = 0
    FP = 0
    FN = 0
    precision = list(); recall = list()

    # to compute mAP
    threshold = 1
    checkpoint = 0
    temp = 1000

    for n, detection in tqdm(enumerate(detections_list)):
        match_flag = False
        if threshold != temp:
            #print(threshold)
            temp = threshold

        # Get groundtruth of the target frame
        gt_on_frame = [x for x in groundtruth_list if x['frame'] == detection['frame']]
        gt_bboxes = [(box(o), o['confidence'] if ('confidence' in o) else 1) for o in gt_on_frame]

        #print(gt_bboxes)
        for gt_bbox in gt_bboxes:
            iou = bb_iou(gt_bbox[0], box(detection))
            if iou > IoU_threshold and gt_bbox[1] > 0.9:
                match_flag = True
                TP += 1
                gt_used = next((x for x in groundtruth_list if x['frame'] == detection['frame'] and box(x) == gt_bbox[0]), None)
                gt_used['confidence'] = 0
                break

        if match_flag == False:
            FP += 1

        # Save metrics
        precision.append(TP/(TP+FP))
        if groundtruth_size:
            recall.append(TP/groundtruth_size)



    recall_step = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.999]
    precision_step = [0] * 11
    max_precision_i = 0.0
    for i in range(len(recall)):
        recall_i = recall[-(i+1)]
        precision_i = precision[-(i+1)]
        max_precision_i = max(max_precision_i, precision_i)
        for j in range(len(recall_step)):
            if recall_i >= recall_step[j]:
                precision_step[j] = max_precision_i
            else:
                break
    if verbose:
        plt.figure(1)
        plt.plot(recall, precision,'r--')
        plt.xlim((0, 1.0))
        plt.ylim((0, 1.0))
        plt.title('Precision - recall curve')
        plt.plot(recall_step, precision_step,'g--')
        plt.show()

    # Check false negatives
    FN = len(detections_list) - TP
    # groups = defaultdict(list)
    # for obj in groundtruth_list:
    #     groups[obj['frame']].append(obj)
    # grouped_groundtruth_list = groups.values()
    #
    # for groundtruth in grouped_groundtruth_list:
    #     detection_on_frame = [x for x in detections_list if x['frame'] == groundtruth[0]['frame']]
    #     detection_bboxes = [box(o) for o in detection_on_frame]
    #
    #     groundtruth_bboxes = [box(o) for o in groundtruth]
    #
    #     results = get_single_frame_results(detection_bboxes, groundtruth_bboxes, IoU_threshold)
    #     FN_temp = results['false_neg']
    #
    #     FN += FN_temp

    if verbose:
        print("TP={} FN={} FP={}".format(TP, FN, FP))

    if TP > 0:
        precision = float(TP) / float(TP + FP)
        recall = float(TP) / float(TP + FN)
        F1_score = 2 * recall * precision / (recall + precision)
    #print(TP+FP)
    #print("precision:{}".format(precision))
    #print("recall:{}".format(recall))
    mAP = sum(precision_step)/11
    #if verbose:
    print("mAP: {}".format(mAP))

    #return precision, recall, precision_step, F1_score, mAP
    return mAP

def read_txt_annotations(gt_path,analyze=False):
    gt_list = {}
    path="../../AIC20_track3/train/S03/c010/gt/gt.txt "
    if analyze:
        max_w = 0
        min_w = 2000
        max_h = 0
        min_h = 2000
        min_ratio = 100
        max_ratio = 0
        
        f= open(path,"r")
        lines = f.readlines()
        for line in lines:
           # print(line)
            l = line.split(',')
            
            gt_list=l.append(Detection(int(l[0])-1, 'car', 
                                             int(float(l[2])), int(float(l[3])), 
                                             int(float(l[4])), int(float(l[5])),
                                             float(l[6]), track_id=int(l[1])))
            if analyze:
                if int(l[4]) < min_w: min_w = int(l[4])
                if int(l[4]) > max_w: max_w = int(l[4])
                if int(l[5]) < min_h: min_h = int(l[5])
                if int(l[5]) > max_h: max_h = int(l[5])
                if int(l[5])/int(l[4]) > max_ratio: max_ratio = int(l[5])/int(l[4])
                if int(l[5])/int(l[4]) < min_ratio: min_ratio = int(l[5])/int(l[4])
    return gt_list
