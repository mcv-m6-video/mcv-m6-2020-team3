import os
import sys
import cv2
import numpy as np
import argparse

# sys.path.append('../MTSC')
# from track_overlap import overlap_tracking, show_tracked_detections
from kalman_tracking import run_track
from task2_utils import compute_idf1
from histogram import compute_histogram, compare_histogram_blocks
import matplotlib.pyplot as plt
# from histogram_tracking import get_detections
import random

parser = argparse.ArgumentParser()
parser.add_argument('--path_sequences', type=str, default='/dataset/aic20-track3-mtmc-train/', help='')
parser.add_argument('--tracking_type', type=str, default='kalman', help='')
parser.add_argument('--visualize', action='store_false', help='')
parser.add_argument('--sequence', type=str, default='train/S01', help='')

prev_colors = {}


def uniqe_color(idx):
    if idx in prev_colors:
        return prev_colors[idx]
    else:
        r = int(random.random() * 256)
        g = int(random.random() * 256)
        b = int(random.random() * 256)
        prev_colors[idx] = (r, g, b)
        return prev_colors[idx]


def visualize_tracking(frames, rois, dets, title='vis'):
    for frame, det in zip(frames, dets):
        if det is not None:
            for box in det:
                x, y, w, h, conf, car_id = box
                cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), \
                              uniqe_color(car_id), 9)
                cv2.rectangle(frame, (int(x), int(y)), (int(x), int(y)), \
                              (0, 0, 0), 15)

    i = 0
    row = None
    ratios = []
    frame_pos = []
    one_frame_cam_num = int(np.ceil(np.sqrt(len(frames))))
    one_frame = None
    placing_pos = [0, 0]

    for frame, det, roi in zip(frames, dets, rois):
        frame_cpy = frame.copy()
        frame[roi == 0] = 0
        alpha = 0.65
        frame = cv2.addWeighted(frame, alpha, frame_cpy, 1 - alpha, 0.0)

        i += 1
        ratios.append([frame.shape[1] / 600, frame.shape[0] / 300])
        frame = cv2.resize(frame, (600, 300))
        if row is None:
            row = frame.copy()
            placing_pos[0] = 0
            frame_pos.append(placing_pos.copy())
        else:
            row = np.concatenate((row, frame), axis=1)
            placing_pos[0] += 600
            frame_pos.append(placing_pos.copy())

        if i % one_frame_cam_num == 0:
            if one_frame is None:
                one_frame = row.copy()
            else:
                one_frame = np.concatenate((one_frame, row), axis=0)
            placing_pos[0] = 0
            placing_pos[1] += 300
            row = None

        if i == len(frames) and (row is not None):
            while i % one_frame_cam_num:
                i += 1
                row = np.concatenate((row, frame * 0), axis=1)
                placing_pos[0] += 600
                frame_pos.append(placing_pos.copy())

            one_frame = np.concatenate((one_frame, row), axis=0)

    for i in range(len(frames)):
        not_connected = True
        for j in range(i + 1, len(frames)):
            dets_1 = dets[i]
            dets_2 = dets[j]
            if dets_1 is not None and dets_2 is not None:
                for box1 in dets_1:
                    for box2 in dets_2:
                        x1, y1, w, h, conf, car_id1 = box1
                        x2, y2, w, h, conf, car_id2 = box2

                        x1 = int((x1) / ratios[i][0] + frame_pos[i][0])
                        x2 = int((x2) / ratios[j][0] + frame_pos[j][0])
                        y1 = int((y1) / ratios[i][1] + frame_pos[i][1])
                        y2 = int((y2) / ratios[j][1] + frame_pos[j][1])

                        if car_id1 == car_id2 and not_connected:
                            cv2.line(one_frame, (x1, y1), (x2, y2),
                                     uniqe_color(car_id1), 3, 8)
                            not_connected = False

    cv2.imshow(title, one_frame)
    cv2.waitKey(1)

    return one_frame


if __name__ == '__main__':
    FLAGS, unparsed = parser.parse_known_args()
    sequence = FLAGS.sequence
    path_sequences = FLAGS.path_sequences
    camera_dirs = os.listdir(path_sequences + sequence)
    camera_dirs.sort()
    video_dirs = []
    video_readers = []
    gt_paths = []
    detections_paths = []
    gt_bboxes = []
    detections_bboxes = []
    roi_imgs = []
    FPS = 10

    for camera_dir in camera_dirs:
        path = path_sequences + sequence + '/' + camera_dir
        video_dirs.append(path + '/vdo.avi')
        gt_paths.append(path + '/gt/gt.txt')
        detections_paths.append(path + '/det/det_mask_rcnn.txt')
        roi = cv2.imread(path + '/roi.jpg', 0)
        _, roi = cv2.threshold(roi, 127, 255, cv2.THRESH_BINARY)
        roi_imgs.append(roi)
        video_readers.append(cv2.VideoCapture(path + '/vdo.avi'))
        gtb, detb = get_detections(path + '/det/det_mask_rcnn.txt', path + '/gt/gt.txt')
        gt_bboxes.append(gtb)
        detections_bboxes.append(detb)

    frame_idx = 0
    while True:  # loop on corresponding frames
        frame_idx += 1
        vis_frames = []
        gt_vis_dets = []
        for cam_no in range(len(video_dirs)):  # loop on cameras
            video_reader = video_readers[cam_no]
            detections_path = detections_paths[cam_no]
            gt_path = gt_paths[cam_no]
            gt_bbox = gt_bboxes[cam_no]
            det_bbox = detections_bboxes[cam_no]

            success, frame = video_reader.read()
            vid_frame_idx = video_reader.get(cv2.CAP_PROP_POS_FRAMES)
            # print(str(vid_frame_idx))
            # print(gt_bbox.keys())
            # print(str(vid_frame_idx) in gt_bbox)
            vis_frames.append(frame)

            if str(int(vid_frame_idx)) in gt_bbox:
                gt_vis_dets.append(gt_bbox[str(int(vid_frame_idx))])
            else:
                gt_vis_dets.append(None)

        one_frame = visualize_tracking(vis_frames, roi_imgs, gt_vis_dets, 'gt_vis')
        cv2.imwrite('/home/sanket/Documents/mcv-m6-2020-team3/week5/output/' + str(frame_idx) + '.jpeg', one_frame)

