import xml.etree.ElementTree as ET
from track import Track
from tqdm import tqdm
import numpy as np
from utils import calculate_mAP
import glob
import cv2
import imageio
import motmetrics as mm
import pickle
from AICityIterator import AICityIterator
from AICityGroundtruth import getGroundtruth


def compute_mAP_track(groundtruth_tracks, detections_tracks, IoU_threshold=0.5, verbose = False):
    """
    Calculate the mean of the best mAP for each track of the ground truth
    input:
    groundtruth_tracks: tracks of gt, the format should be list of Class Track
    detections_tracks: tracks of detections, the format should be list of Class Track
    """

    mAP_list = list()

    for groundtruth_track in tqdm(groundtruth_tracks):
        #print('DETECTION'), print(detection_track)
        max_mAP = 0

        for detection_track in detections_tracks:
            #print(groundtruth_track)
            mAP = calculate_mAP(groundtruth_track.detections, detection_track.detections,
                                          IoU_threshold)
            #print(mAP)
            if(max_mAP < mAP):
                max_mAP = mAP

        mAP_list.append(max_mAP)

    np_mAP_list = np.asarray(mAP_list)

    mAP_mean = np.mean(np_mAP_list)
    if verbose:
        print("np_mAP_list:")
        print(np_mAP_list)
        print("mAP = {}".format(mAP_mean))

    return mAP_mean


def center_of_detection(detection):
    return (int(detection['left'] + 0.5 * detection['width']), int(detection['top'] + 0.5 * detection['height']))


def write_one_frame(detections_tracks, frame_id, frameMat, color):
    """
    tool for addTracksToFrames
    :param detections_tracks: this can be detections or ground truth
    :param frame_id: which frame
    :param frameMat: frame picture
    :param color: rectangle and line color
    :return: frameMat
    """
    for track_one in detections_tracks:
        index = 0
        flag_shoot = False
        for index, detection in enumerate(track_one.detections):
            # write the rectangle
            if detection['frame'] == frame_id:
                startPoint = (int(detection['left']), int(detection['top']))
                endPoint = (int(startPoint[0] + detection['width']), int(startPoint[1] + detection['height']))
                frameMat = cv2.rectangle(frameMat, startPoint, endPoint, color, 2)
                flag_shoot = True
                break
        if flag_shoot:
            shoot_index = index
            # write the line
            for index in range(shoot_index - 1):
                startPoint = center_of_detection(track_one.detections[index])
                endPoint = center_of_detection(track_one.detections[index + 1])
                frameMat = cv2.line(frameMat, startPoint, endPoint, color, 2)
    return frameMat


def addTracksToFrames(framesPath, detections_tracks, tracks_gt_list, start_frame = 1, end_frame = 2141, name = "test"):
    """
    write the video of the tracking result in the format .avi
    :param framesPath: path of frames
    :param detections_tracks: detections in list of Track. Detections in Track should be sorted.
    :param tracks_gt_list: ground truth in list of Track. Detections in Track should be sorted.
    :param name: name of video.
    :return: None
    """
    size = (1920, 1080)
    fps = 10
    out = cv2.VideoWriter(name + '.avi', cv2.VideoWriter_fourcc(*'DIVX'), fps, size)

    for frame_id in tqdm(range(start_frame, end_frame)):
        filename = "{}{}.jpg".format(framesPath, str(frame_id).zfill(5))
        frameMat = cv2.imread(filename)
        color_detection = (0, 0, 255)
        write_one_frame(detections_tracks, frame_id, frameMat, color_detection)
        color_gt = (255, 0, 0)
        write_one_frame(tracks_gt_list, frame_id, frameMat, color_gt)
        out.write(frameMat)
    out.release()


def addTracksToFrames_gif(framesPath, detections_tracks, tracks_gt_list, start_frame = 1, end_frame = 10, name = "test"):
    """
    write the gif of the tracking result
    :param framesPath: path of frames
    :param detections_tracks: detections in list of Track. Detections in Track should be sorted.
    :param tracks_gt_list: ground truth in list of Track. Detections in Track should be sorted.
    :param name: name of video.
    :return: None
    """
    images = []

    for frame_id in tqdm(range(start_frame, end_frame)):
        filename = "{}{}.jpg".format(framesPath, str(frame_id).zfill(5))
        frameMat = cv2.imread(filename)
        color_detection = (0, 0, 255)
        write_one_frame(detections_tracks, frame_id, frameMat, color_detection)
        color_gt = (255, 0, 0)
        write_one_frame(tracks_gt_list, frame_id, frameMat, color_gt)
        resized = cv2.resize(frameMat, (480, 270), interpolation=cv2.INTER_AREA)
        images.append(resized)
    imageio.mimsave(name + '.gif', images)




def find_frame_in_track(tracks, frame_id):
    object_id_list = []
    box_list = []
    for track_one in tracks:
        for index, detection in enumerate(track_one.detections):
            # write the rectangle
            if detection['frame'] == frame_id:
                object_id_list.append(track_one.id)
                box_list.append([detection['left'], detection['top'], detection['width'], detection['height']])
                break
    return object_id_list, box_list

def find_frame_in_gt(gt, frame_id):
    object_id_list = []
    box_list = []
    for instance in gt:
        if instance['frame'] == frame_id:
            object_id_list.append(instance['ID'])
            box_list.append([instance['left'], instance['top'], instance['width'], instance['height']])
    return object_id_list, box_list


def calculate_idf1(gt, detections_tracks, video_length, IoU_threshold=0.5, verbose = False):
    acc = mm.MOTAccumulator(auto_id=True)

    for i in range(video_length):
        frame_id = i + 1
        gt_ids, gt_bboxes = find_frame_in_gt(gt, frame_id)
        detections_ids, detections_bboxes = find_frame_in_track(detections_tracks, frame_id)

        distances_gt_det = mm.distances.iou_matrix(gt_bboxes, detections_bboxes, max_iou=1.)
        acc.update(gt_ids, detections_ids, distances_gt_det)

    print(acc.mot_events)
    mh = mm.metrics.create()
    summary = mh.compute(acc, metrics=mm.metrics.motchallenge_metrics, name='acc')
    print(summary)



def tracking_filter(detections_tracks, frameDistance=10, trackLength=100):
    detections_tracks_filtered = []
    for track_one in detections_tracks:
        detection_first = track_one.detections[0]
        X_1 = detection_first['left'] + 0.5*detection_first['width']
        Y_1 = detection_first['top'] + 0.5*detection_first['height']
        detection_last = track_one.detections[-1]
        X_2 = detection_last['left'] + 0.5 * detection_last['width']
        Y_2 = detection_last['top'] + 0.5 * detection_last['height']
        vec1 = np.array([X_1, Y_1])
        vec2 = np.array([X_2, Y_2])
        if (detection_last['frame']-detection_first['frame']) > frameDistance:
            if np.linalg.norm(vec1-vec2) > trackLength:
                detections_tracks_filtered.append(track_one)
    return detections_tracks_filtered

def read_tracking_annotations(seq, cam, video_length=None):
    tracks = {}
    groundtruthList = []
    tracksGtList = []
    iterator = AICityIterator(seq, cam, video_length)
    for i, _ in tqdm(enumerate(iterator), total=len(iterator)):
        gt = getGroundtruth(seq, cam, i+1)
        for item in gt:
            groundtruthList.append(item)
            if item['ID'] not in tracks.keys():
                tracks[item['ID']] = Track(item['ID'], [item])
            else:
                tracks[item['ID']].addDetection(item)
    for id in sorted(tracks.keys()):
        tracksGtList.append(tracks[id])
    return groundtruthList, tracksGtList


if __name__ == "__main__":
    pass


