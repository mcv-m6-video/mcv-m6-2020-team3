from __future__ import print_function
import os
import argparse
import pickle

import numpy as np
import cv2
from sort import Sort
from task2_utils import read_xml_annotations, compute_idf1, check_size_bbox

parser   = argparse.ArgumentParser()

parser.add_argument('--video_dir', type=str, default='../datasets/AICity_data/train/S03/c010/vdo.avi', help='')
parser.add_argument('--labels_dir', type=str, default='rcnn_detections_before_tuning/', help='')
parser.add_argument('--annotations_dir', type=str, default='./annotations/', help='')

classes_of_interest = ['car', 'bus', 'truck']
min_confidence      = 0.65


def run_track(video_dir, dict_detections, detector, size_threshold=8000, visualize=False, wait_time=0, get_first_appearance=False):

    capture = cv2.VideoCapture(video_dir)

    frame_idx = 0
    kalman_tracker = Sort()

    whole_video_detections = []
    first_appearance = dict()
    while True:
        success, frame = capture.read()
        if not success:
            break
        frame_idx += 1

        detections = dict_detections.get(str(frame_idx))
        to_track = []
        if detections is not None:
            for bbox in detections:
                if len(bbox) != 0:
                    x = int(bbox[0])
                    y = int(bbox[1])
                    w = int(bbox[2])
                    h = int(bbox[3])
                    size_bbox = check_size_bbox([x,y,w,h])
                    if size_bbox > size_threshold:
                        to_append = [x,y,x+w,y+h]
                        to_track.append(to_append)

        trackers = kalman_tracker.update(np.array(to_track))

        current_frame_detections = []
        for track_det in trackers:
            track_det = track_det.astype(np.int64)
            current_frame_detections.append(['car', track_det[0], track_det[1], track_det[2],
                                             track_det[3], track_det[4]])

            if get_first_appearance:
                if str(int(track_det[4])) not in first_appearance.keys():
                    first_appearance[str(int(track_det[4]))] = ['car', track_det[0], track_det[1],
                                                               track_det[2], track_det[3], capture.get(cv2.CAP_PROP_POS_FRAMES)]

            if visualize:
                x = int(track_det[0])
                y = int(track_det[1])
                w = int(track_det[2])
                h = int(track_det[3])
                # if y < frame.shape[1] and h<frame.shape[1] and x<frame.shape[0] and w<frame.shape[0]:
                cv2.rectangle(frame, (x, y), (w, h), (0, 0, 255), 3)

                font = cv2.FONT_HERSHEY_DUPLEX
                placement = (w + 10, h + 10)
                font_scale = 1
                font_color = (0, 255, 0)
                line_type = 2

                cv2.putText(frame, str(track_det[4]), placement, font, font_scale, font_color, line_type)

        if visualize:
            cv2.namedWindow('output', cv2.WINDOW_NORMAL)
            cv2.imshow('output', frame)
            cv2.waitKey(wait_time)

        whole_video_detections.append(current_frame_detections)

    if get_first_appearance:
        return whole_video_detections, first_appearance

    return whole_video_detections





