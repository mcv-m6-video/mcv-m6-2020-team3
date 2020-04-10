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


if __name__ == "__main__":
    detections_filename = "../detections/detections_all_camera_fine_tune.pkl"
    test_path = "../Datasets/AIC20_track3/train/S03/"

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

        detections_tracks = find_tracking_maximum_overlap(detections, video_length, missing_chance=missing_chance,
                                                          lou_max_threshold=lou_max_threshold)

        # filter the track
        detections_tracks = tracking_filter(detections_tracks)

        # mAP_track = compute_mAP_track(tracks_gt_list, detections_tracks, IoU_threshold=0.5)
        # print("mAP_track = ", mAP_track)

        calculate_idf1(gt, detections_tracks, video_length)

        for track_one in detections_tracks:
            track_one.detections.sort(key=lambda x: x['frame'])

        addTracksToFrames('{}{}/frames/'.format(test_path, camera), detections_tracks, tracks_gt_list,
                          start_frame=1, end_frame=video_length, name="test_track"+camera)
        # addTracksToFrames_gif(video_path, detections_tracks, tracks_gt_list, start_frame=210, end_frame=390, name="test")




