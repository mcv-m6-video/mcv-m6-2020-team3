#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 00:12:37 2020

@author: kaiali
"""

def compute_iou(gt_box,b_box):
    '''
    compute iou
    :param gt_box: ground truth gt_box = [x0,y0,x1,y1]（x0,y0)left-up（x1,y1）right-down
    :param b_box: bounding box b_box--same as gt_box
    :return: 
    '''
    width0= abs(gt_box[2]-gt_box[0])
    height0 = abs(gt_box[3] - gt_box[1])
    
    width1 = abs(b_box[2] - b_box[0])
    height1 = abs(b_box[3] - b_box[1])
    
    max_x =max(gt_box[2],b_box[2])
    min_x = min(gt_box[0],b_box[0])
    width_iou = width0 + width1 -(max_x-min_x)
    
    max_y = max(gt_box[3],b_box[3])
    min_y = min(gt_box[1],b_box[1])
    height_iou = height0 + height1 - (max_y - min_y)
 
    interArea = width_iou * height_iou
    boxAArea = width0 * height0
    boxBArea = width1 * height1
    iou = interArea / float(boxAArea + boxBArea - interArea)
    
    return iou
 
