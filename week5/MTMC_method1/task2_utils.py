import copy

from random import randrange, random
import numpy as np
from xml.dom import minidom
import os
import motmetrics as mm

def bbox_iou(src_bboxA, src_bboxB):
    # compute the intersection over union of two bboxes
    # I've adapted this code from the M1 base code. The function expects [tly, tlx, width, height],
    # where tl indicates the top left corner of the box.
    bboxA = copy.deepcopy(src_bboxA)
    bboxB = copy.deepcopy(src_bboxB)

    if len(bboxB) == 0 or len(bboxA) == 0:
        return 0.0, [0, 0, 0, 0]
    bboxA[2] = bboxA[0] + bboxA[2]
    bboxA[3] = bboxA[1] + bboxA[3]
    bboxB[2] = bboxB[0] + bboxB[2]
    bboxB[3] = bboxB[1] + bboxB[3]

    xA = max(bboxA[1], bboxB[1])
    yA = max(bboxA[0], bboxB[0])
    xB = min(bboxA[3], bboxB[3])
    yB = min(bboxA[2], bboxB[2])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both bboxes
    bboxAArea = (bboxA[2] - bboxA[0] + 1) * (bboxA[3] - bboxA[1] + 1)
    bboxBArea = (bboxB[2] - bboxB[0] + 1) * (bboxB[3] - bboxB[1] + 1)

    try:
        iou = interArea / float(bboxAArea + bboxBArea - interArea)
    except ZeroDivisionError:
        iou = 0.0

    # return the intersection over union value
    return iou, src_bboxB

def read_xml_annotations(annotations_path):
    files = os.listdir(annotations_path)

    bboxes = dict()
    for file_ in files:
        xmldoc = minidom.parse(annotations_path + file_)
        bboxes_list = xmldoc.getElementsByTagName('box')
        for element in bboxes_list:
            frame = element.getAttribute('frame')
            xtl = int(float(element.getAttribute('xtl')))
            ytl = int(float(element.getAttribute('ytl')))
            xbr = int(float(element.getAttribute('xbr')))
            ybr = int(float(element.getAttribute('ybr')))
            height = ybr - ytl
            width  = xbr - xtl
            id_det = int(element.parentNode.getAttribute('id'))

            if frame in bboxes.keys():
                bboxes[frame].append(['car', xtl, ytl, height, width, random(), id_det])
            else:
                bboxes[frame] = [['car', xtl, ytl, height, width, random(), id_det]]
    return bboxes


def compute_idf1(list_gt_bboxes, tracked_detections):
    """The lists passed are of the form [label, tly, tlx, width, height, confidence, id]
    gt is of the form [frame, id, tlx, tly, width, height, confidence]
    """
    acc = mm.MOTAccumulator(auto_id=True)
    for gt_elements_frame, det_elements_frame in zip(list_gt_bboxes, tracked_detections):
        det_ids = []
        gt_ids = []
        mm_det_bboxes = []
        mm_gt_bboxes = []

        for gt_bbox in gt_elements_frame:
            if len(gt_bbox) != 0:
                # mm_gt_bboxes.append([(gt_bbox[1]+gt_bbox[3])/2, (gt_bbox[2]+gt_bbox[4])/2, gt_bbox[3]-gt_bbox[1],
                #                      gt_bbox[4]-gt_bbox[2]])
                mm_gt_bboxes.append([gt_bbox[2], gt_bbox[1], gt_bbox[3], gt_bbox[4]])

                gt_ids.append(gt_bbox[-1])
            else:
                mm_gt_bboxes.append([None, None, None, None])
                gt_ids.append(None)

        for det_bbox in det_elements_frame:
            if len(det_bbox) != 0:
                # mm_det_bboxes.append([(det_bbox[1]+det_bbox[3])/2, (det_bbox[2]+det_bbox[4])/2, det_bbox[3]-det_bbox[1],
                #                      det_bbox[4]-det_bbox[2]])
                mm_det_bboxes.append([det_bbox[2], det_bbox[1], det_bbox[3], det_bbox[4]])

                det_ids.append(det_bbox[-1])
            else:
                mm_det_bboxes.append([None, None, None, None])
                det_ids.append(None)

        distances_gt_det = mm.distances.iou_matrix(mm_gt_bboxes, mm_det_bboxes, max_iou=1.)
        acc.update(gt_ids, det_ids, distances_gt_det)

    mh = mm.metrics.create()
    summary = mh.compute(acc, metrics=['idf1'], name='acc')
    return summary


def check_size_bbox(bbox):
    """bbox = x,y,w,h"""
    x = bbox[0]
    y = bbox[1]
    w = bbox[2]
    h = bbox[3]
    return w*h