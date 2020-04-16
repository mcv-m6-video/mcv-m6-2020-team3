import pickle
from distance_metric_learning import train_metric
from utils_evaluation import compute_idf1
from features import compute_mr_histogram
import random
import cv2
import numpy as np


def crop_from_detection(det):
    image = cv2.imread(det['img_path'])
    box = [int(det['left']), int(det['top']), int(det['width']), int(det['height'])]
    cropped = image[box[1]:box[1]+box[3], box[0]:box[0]+box[2], :]
    return cropped

if __name__ == "__main__":
    frame_path_S3 = '../../Datasets/AIC20_track3/train/S03/'
    camera_list = ['c010', 'c011', 'c012', 'c013', 'c014', 'c015']
    timestamp = {
        'c010': 8.715,
        'c011': 8.457,
        'c012': 5.879,
        'c013': 0,
        'c014': 5.042,
        'c015': 8.492,
    }
    for key in timestamp.keys():
        timestamp[key] = int(timestamp[key])

    # The frame rates of all the videos are 10 FPS, except for c015 in S03, whose frame rate is 8 FPS.
    fps_ratio = {
        'c010': 1.0,
        'c011': 1.0,
        'c012': 1.0,
        'c013': 1.0,
        'c014': 1.0,
        # 'c015': 10.0 / 8.0,
        'c015': 1.0,
    }

    video_length_list = {
        'c010': 2141,
        'c011': 2279,
        'c012': 2422,
        'c013': 2415,
        'c014': 2332,
        'c015': 1928}

    offset = {
        'c010': 0,
        'c011': 3000,
        'c012': 6000,
        'c013': 9000,
        'c014': 12000,
        'c015': 15000}

    # read the result of tracing with single camera.
    with open("detections_tracks_all_camera.pkl", 'rb') as f:
        detections_tracks_all_camera = pickle.load(f)
        f.close()
    with open("gt_tracks_all_camera.pkl", 'rb') as f:
        gt_tracks_all_camera = pickle.load(f)
        f.close()

    # transform the tracks of detections, sync the time frame.
    detections_tracks_all_camera_list = []
    idx = 0
    for cam in detections_tracks_all_camera.keys():
        for track_one in detections_tracks_all_camera[cam]:
            track_one.id = idx
            idx = idx + 1
            for detection in track_one.detections:
                detection['img_path'] = "{}{}/frames/{}.jpg".format(frame_path_S3, cam,
                                                                    str(detection['frame']).zfill(5))
                detection['cam'] = cam
                detection['frame'] = int(detection['frame'] * fps_ratio[cam] + timestamp[cam])

            detections_tracks_all_camera_list.append(track_one)

    gt_tracks_all_camera_list = []
    for cam in gt_tracks_all_camera.keys():
        for track_one in gt_tracks_all_camera[cam]:
            for detection in track_one.detections:
                detection['img_path'] = "{}{}/frames/{}.jpg".format(frame_path_S3, cam,
                                                                    str(detection['frame']).zfill(5))
                detection['cam'] = cam
                detection['frame'] = int(detection['frame'] * fps_ratio[cam] + timestamp[cam])
            gt_tracks_all_camera_list.append(track_one)

    compute_idf1(gt_tracks_all_camera_list, detections_tracks_all_camera_list, 2422)

    feature_function = lambda img: compute_mr_histogram(img, splits=(3, 3), bins=32,
                                                        mask=None, sqrt=False, concat=True)

    nca = train_metric(feature_function)

    num_compare_pairs = 10

    for cam_index, cam1 in enumerate(camera_list):
        for track_one in detections_tracks_all_camera[cam1]:
            for cam2 in camera_list[cam_index + 1:]:
                for track_another in detections_tracks_all_camera[cam2]:
                    pairs = []
                    for i in range(num_compare_pairs):
                        det1 = random.sample(track_one.detections, 1)[0]
                        cropped = crop_from_detection(det1)
                        vec1 = feature_function(cropped)
                        det2 = random.sample(track_another.detections, 1)[0]
                        cropped = crop_from_detection(det2)
                        vec2 = feature_function(cropped)
                        pairs.append(np.vstack((vec1, vec2)))
                    pairs = np.array(pairs)
                    scores = nca.score_pairs(pairs).mean()
                    if scores < 50.0:
                        track_another.id = track_one.id

    detections_tracks_all_camera_list = []
    for cam in detections_tracks_all_camera.keys():
        for track_one in detections_tracks_all_camera[cam]:
            detections_tracks_all_camera_list.append(track_one)

    gt_tracks_all_camera_list = []
    for cam in gt_tracks_all_camera.keys():
        for track_one in gt_tracks_all_camera[cam]:
            gt_tracks_all_camera_list.append(track_one)

    compute_idf1(gt_tracks_all_camera_list, detections_tracks_all_camera_list, 2422)




