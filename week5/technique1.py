"""
example
{'frame': 100, 'left': 931, 'top': 78, 'width': 82, 'height': 68, 'confidence': 0.99}
Note, frame starts from 1.
"""
import pickle
from utils import addBboxesToFrames_avi, calculate_mAP, bb_iou, addBboxesToFrames_gif, upscaleDetections
from utils_tracking import tracking_filter, compute_mAP_track, calculate_idf1, addTracksToFrames, addTracksToFrames_gif
from utils_read import read_gt_txt, transform_gt
from utils_maximum_overlap import find_tracking_maximum_overlap

from sort import Sort
import numpy as np

from utils_kalman import kalman_filter_tracking_2


if __name__ == "__main__":
    # detections_filename = "../detections/detections_faster_rcnn_pre_trained_test_S03.pkl"
    detections_filename = "../detections/detections_faster_rcnn_fine_tune_test_S03.pkl"
    test_path = "../Datasets/AIC20_track3/train/S03/"

    track_type = "maximum overlap"
    # track_type = "kalman filter"
    # kalman_filter_mode = 1

    filter_track = True

    print("Reading pkl")
    with open(detections_filename, 'rb') as p:
        detections_all_camera = pickle.load(p)
        p.close()

    camera_list = ['c010', 'c011', 'c012', 'c013', 'c014', 'c015']
    video_length_list = {
        'c010': 2141,
        'c011': 2279,
        'c012': 2422,
        'c013': 2415,
        'c014': 2332,
        'c015': 1928}

    idf1_list = []
    detections_tracks_all_camera = {}
    for camera in camera_list:
        print(camera)

        detections = detections_all_camera[camera]
        print("Reading gt...")
        gt = read_gt_txt('{}{}/gt/gt.txt'.format(test_path, camera))
        tracks_gt_list = transform_gt(gt)

        print("calculate mAP...")
        mAP = calculate_mAP(gt, detections, IoU_threshold=0.5, have_confidence=True, verbose=True)
        print("mAP = ", mAP)

        # addBboxesToFrames_avi('{}{}/frames'.format(test_path, camera), detections, gt, "test")
        # addBboxesToFrames_gif(video_path, detections, groundTruth, start_frame=210, end_frame=260, name="test")

        # sort detections for following operations.
        detections.sort(key=lambda x: x['frame'])

        #calculate video_length
        video_length = video_length_list[camera]

        missing_chance = 5
        lou_max_threshold = 0.5

        if track_type == "maximum overlap":
            # maximum overlap
            detections_tracks = find_tracking_maximum_overlap(detections, video_length, missing_chance=missing_chance,
                                                              lou_max_threshold=lou_max_threshold)
        elif track_type == "kalman filter":
            # kalman
            # mode: 0: velocity; 1: acceleration
            detections_tracks = kalman_filter_tracking_2(detections, video_length, kalman_filter_mode)
        else:
            raise Exception("Wrong track type")

        if filter_track:
            # filter the track
            detections_tracks = tracking_filter(detections_tracks)

        # mAP_track = compute_mAP_track(tracks_gt_list, detections_tracks, IoU_threshold=0.5)
        # print("mAP_track = ", mAP_track)

        idf1 = calculate_idf1(gt, detections_tracks, video_length)
        idf1_list.append(idf1)

        for track_one in detections_tracks:
            track_one.detections.sort(key=lambda x: x['frame'])

        # addTracksToFrames('{}{}/frames/'.format(test_path, camera), detections_tracks, tracks_gt_list,
        #                   start_frame=1, end_frame=video_length, name="test_track"+camera)
        # addTracksToFrames_gif(video_path, detections_tracks, tracks_gt_list, start_frame=210, end_frame=390, name="test")

        detections_tracks_all_camera[camera] = detections_tracks
    print(idf1_list)
    with open("detections_tracks_all_camera.pkl", 'wb') as f:
        pickle.dump(detections_tracks_all_camera, f)
        f.close()

