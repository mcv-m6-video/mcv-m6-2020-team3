import pickle
from distance_metric_learning import train_metric
from utils_evaluation import compute_idf1
from features import compute_mr_histogram
import random
import cv2
import numpy as np
from tqdm import tqdm
import copy
from utils_mul_cam import merge_tracks, addTracksToFrames_multi_cam, sort_track, addTracksToFrames_multi_cam_gif


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
        'c015': 10.0 / 8.0,
    }

    video_length_list = {
        'c010': 2141,
        'c011': 2279,
        'c012': 2422,
        'c013': 2415,
        'c014': 2332,
        'c015': 1928}

    offset = {
        'c010': [6000, 2000],
        'c011': [6000, 0],
        'c012': [3000, 0],
        'c013': [0, 2000],
        'c014': [3000, 2000],
        'c015': [0, 0]}

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
    for cam in camera_list:
        sort_track(detections_tracks_all_camera[cam])
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
    for cam in camera_list:
        sort_track(gt_tracks_all_camera[cam])
        for track_one in gt_tracks_all_camera[cam]:
            for detection in track_one.detections:
                detection['img_path'] = "{}{}/frames/{}.jpg".format(frame_path_S3, cam,
                                                                    str(detection['frame']).zfill(5))
                detection['cam'] = cam
                detection['frame'] = int(detection['frame'] * fps_ratio[cam] + timestamp[cam])
            gt_tracks_all_camera_list.append(track_one)

    compute_idf1(gt_tracks_all_camera_list, detections_tracks_all_camera_list, 2422)

    # define the feature function
    feature_function = lambda img: compute_mr_histogram(img, splits=(3, 3), bins=32,
                                                        mask=None, sqrt=False, concat=True)

    load_nca_pkl = True
    if load_nca_pkl:
        with open("nca.pkl", 'rb') as f:
            nca = pickle.load(f)
            f.close()
    else:
        nca = train_metric(feature_function)
        with open("nca.pkl", 'wb') as f:
            pickle.dump(nca, f)
            f.close()

    load_result_tracks = True
    if load_result_tracks:
        with open("detections_tracks_all_camera_list.pkl", 'rb') as f:
            detections_tracks_all_camera_list = pickle.load(f)
            f.close()
    else:
        num_compare_pairs = 2

        for cam_index, cam1 in enumerate(camera_list):
            for track_one in tqdm(detections_tracks_all_camera[cam1]):
                for cam2 in camera_list[cam_index + 1:]:
                    for track_another in detections_tracks_all_camera[cam2]:
                        pairs_cross = []
                        det1 = track_one.detections[int(0.5*len(track_one.detections))]
                        cropped = crop_from_detection(det1)
                        vec1 = feature_function(cropped)
                        det2 = track_another.detections[int(0.5*len(track_another.detections))]
                        cropped = crop_from_detection(det2)
                        vec2 = feature_function(cropped)
                        pairs_cross.append(np.vstack((vec1, vec2)))
                        pairs_cross = np.array(pairs_cross)
                        score_cross = nca.score_pairs(pairs_cross).mean()
                        if score_cross < 60:
                            track_another.id = track_one.id

        detections_tracks_all_camera_list = []
        for cam in detections_tracks_all_camera.keys():
            for track_one in detections_tracks_all_camera[cam]:
                detections_tracks_all_camera_list.append(track_one)

        with open("detections_tracks_all_camera_list.pkl", 'wb') as f:
            pickle.dump(detections_tracks_all_camera_list, f)
            f.close()

    new_detections_tracks = merge_tracks(detections_tracks_all_camera_list)

    gt_tracks_all_camera_list = []
    for cam in gt_tracks_all_camera.keys():
        for track_one in gt_tracks_all_camera[cam]:
            gt_tracks_all_camera_list.append(track_one)
    new_gt_tracks = merge_tracks(gt_tracks_all_camera_list)

    sort_track(new_detections_tracks)
    sort_track(new_gt_tracks)

    compute_idf1(gt_tracks_all_camera_list, detections_tracks_all_camera_list, 2422)

    # addTracksToFrames_multi_cam(frame_path_S3, new_detections_tracks, new_gt_tracks, offset, camera_list,
    #                             timestamp, fps_ratio, video_length_list, start_frame=1300, end_frame=1600,
    #                             name="test_track")
    addTracksToFrames_multi_cam_gif(frame_path_S3, new_detections_tracks, new_gt_tracks, offset, camera_list,
                                timestamp, fps_ratio, video_length_list, start_frame=1440, end_frame=1550,
                                name="test_track")



