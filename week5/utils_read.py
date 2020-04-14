import os
from track import Track

def read_gt_txt(gt_file):
    """
    This function read the gt in txt.
    :param gt_file:
    :return:
    """
    gt = []
    for line in open(gt_file):
        fields = line.split(",")
        instance = {}
        instance['frame'] = int(fields[0])
        instance['ID'] = int(fields[1])
        instance['left'] = int(fields[2])
        instance['top'] = int(fields[3])
        instance['width'] = int(fields[4])
        instance['height'] = int(fields[5])

        gt.append(instance)
    return gt

def transform_gt(gt):
    tracks_all = []
    for instance in gt:
        flag_not_allocated = True
        for track in tracks_all:
            if track.id == instance['ID']:
                flag_not_allocated = False
                detection = detection_is_dictionary(instance['frame'], instance['left'], instance['top'],
                                                    instance['width'], instance['height'], confidence = 1.0)
                track.detections.append(detection)
        if flag_not_allocated:
            detection = detection_is_dictionary(instance['frame'], instance['left'], instance['top'],
                                                instance['width'], instance['height'], confidence=1.0)
            track_one = Track(instance['ID'], [detection])
            tracks_all.append(track_one)
    return tracks_all

def transform_det(bb_id_updated):
    tracks_all = []
    for instance in bb_id_updated:
        flag_not_allocated = True
        for track in tracks_all:
            if track.id == instance['ID']:
                flag_not_allocated = False
                detection = detection_is_dictionary(instance['frame'], instance['left'], instance['top'],
                                                    instance['width'], instance['height'],
                                                    confidence=instance['confidence'])
                track.detections.append(detection)
        if flag_not_allocated:
            detection = detection_is_dictionary(instance['frame'], instance['left'], instance['top'],
                                                instance['width'], instance['height'],
                                                confidence=instance['confidence'])
            track_one = Track(instance['ID'], [detection])
            tracks_all.append(track_one)
    return tracks_all

def detection_is_dictionary(frame_id, left, top, width, height, confidence = 1.0):
    detection = {}
    detection['frame'] = frame_id
    detection['left'] = left
    detection['top'] = top
    detection['width'] = width
    detection['height'] = height
    detection['confidence'] = confidence
    return detection

