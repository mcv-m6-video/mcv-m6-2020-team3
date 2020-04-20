import os
import sys
import cv2
import numpy as np
import argparse
from kalman_tracking import run_track
# sys.path.insert(0, '../MTSC')
from histogram import compute_histogram,compare_histogram_blocks
from mot import compute_scores
from tracking import visualize_tracking
import sys


parser   = argparse.ArgumentParser()
parser.add_argument('--tracking_type', type=str, default='kalman', help='')
parser.add_argument('--visualize', action='store_false', help='')
path_sequences = '/home/sanket/Documents/mcv-m6-2020-team3/Datasets/AIC20_track3/'


def visualize(detections, video_paths, rois):
    video_readers = []
    roi_imgs = []
    for path, roi_path in zip(video_paths, rois):
        video_readers.append(cv2.VideoCapture(path))
        roi_imgs.append(cv2.imread(roi_path, 0))

    lengths = []
    for video_reader in video_readers:
        length = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
        lengths.append(length)
    min_num_frames = min(lengths)

    count = 0

    for i in range(min_num_frames):
        vis_frames = []
        vis_dets = []

        for video_reader, detections_frame in zip(video_readers, detections):
            _, frame = video_reader.read()
            vis_frames.append(frame)
            detections_formatted = []
            if len(detections_frame[i]) != 0:
                for detection in detections_frame[i]:
                    detections_formatted.append([detection[1],
                                                 detection[2],
                                                 detection[3] - detection[1],
                                                 detection[4] - detection[2],
                                                 1, int(detection[5])])
            else:
                detections_formatted = None
            vis_dets.append(detections_formatted)

        _ = visualize_tracking(vis_frames, roi_imgs, vis_dets, 'det_vis')
        count+=1
        if count == 500:
            break


def get_detections(detections_path, gt_file_path, min_area=None):

    with open(gt_file_path) as f:
        gt_data = f.readlines()

    # Read and prepare groundtruth data
    dict_gt_bboxes = dict()
    for line in gt_data:
        splits = line.split(",")
        frame_idx = splits[0]
        id = int(splits[1])
        x = float(splits[2])
        y = float(splits[3])
        w = float(splits[4])
        h = float(splits[5])
        conf = float(splits[6])
        bbox = [x,y,w,h,conf,id]
        if frame_idx in dict_gt_bboxes.keys():
            dict_gt_bboxes[frame_idx].append(bbox)
        else:
            dict_gt_bboxes[frame_idx] = [bbox]

    with open(detections_path) as f:
        detections_data = f.readlines()

    detections_dict = dict()
    for line in detections_data:
        splits = line.split(",")
        frame_idx = splits[0]
        x = float(splits[2])
        y = float(splits[3])
        w = float(splits[4])
        h = float(splits[5])
        conf = float(splits[6])
        bbox = [x,y,w,h,conf]
        if (w*h) > 7000 and conf > 0.6:
            if frame_idx in detections_dict.keys():
                detections_dict[frame_idx].append(bbox)
            else:
                detections_dict[frame_idx] = [bbox]

    return dict_gt_bboxes, detections_dict


if __name__ == "__main__":
    for sequence in ['train/S04']:
        camera_dirs = os.listdir(path_sequences + sequence)
        reference_histograms = None
        tracks = []
        video_paths = []
        roi_paths = []

        print("sequence: " + sequence)
        for camera_dir in camera_dirs:
            path = path_sequences + sequence + '/' + camera_dir
            roi_paths.append(path + '/roi.jpg')
            video_path = path + '/vdo.avi'
            gt_path = path + '/gt/gt.txt'
            detections_path = path + '/det/det_mask_rcnn.txt'
            video_paths.append(video_path)

            gt_bboxes, detections_bboxes = get_detections(detections_path, gt_path)

            # kalman tracker works better since we have more occlusions
            track, first_detections = run_track(video_path, detections_bboxes, False, wait_time=50, get_first_appearance=True)

            # here we create a dict, the id of the car is the key and the value is the histogram. This is done for the
            # first sequence to have a reference
            if reference_histograms is None:
                print("computing reference hist")
                capture = cv2.VideoCapture(video_path)
                reference_histograms = dict()
                for key in first_detections.keys():
                    bbox = first_detections[key][1:5]
                    frame_num = first_detections[key][5]
                    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                    _, frame = capture.read()
                    patch = frame[max(0, bbox[1]):max(0, bbox[3]), max(0, bbox[0]):max(0, bbox[2]), :]
                    reference_histograms[key] = np.array(compute_histogram(patch))

            # we compare the histograms of the detected cars in the other sequences with each detection from the
            # first sequence, then we reassign the id of the detections to the most similar to the reference
            else:
                print("computing the other hists")
                capture = cv2.VideoCapture(video_path)
                id_correspondence = dict()
                for key in first_detections.keys():
                    try:
                        bbox = first_detections[key][1:5]
                        frame_num = first_detections[key][5]
                        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                        _, frame = capture.read()
                        patch = frame[max(0, bbox[1]):max(0, bbox[3]), max(0, bbox[0]):max(0, bbox[2]), :]

                        min_score = 999999999999
                        new_id = -1
                        for key2 in reference_histograms:
                            current_hist = np.array(compute_histogram(patch))
                            score = compare_histogram_blocks(reference_histograms[key2], current_hist, method=1)
                            if score < min_score:
                                min_score = score
                                new_id = key2
                        id_correspondence[str(key)] = new_id
                    except:  # because fuck you opencv
                        id_correspondence[str(key)] = key

                # reassign the car id to the most similar one
                for i, frame in enumerate(track):
                    for j, detection in enumerate(track[i]):
                        track[i][j][5] = id_correspondence[str(frame[j][5])]

            tracks.append(track)

            compute_scores(gt_bboxes, track)

        visualize(tracks, video_paths, roi_paths)




