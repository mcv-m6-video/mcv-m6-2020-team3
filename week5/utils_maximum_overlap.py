"""
example
{'frame': 100, 'left': 931, 'top': 78, 'width': 82, 'height': 68, 'confidence': 0.99}
Note, frame starts from 1.
"""

from utils import addBboxesToFrames_avi, calculate_mAP, bb_iou, addBboxesToFrames_gif, upscaleDetections
from tqdm import tqdm
from track import Track


def detection_to_box(detection):
    return [detection['left'], detection['top'], detection['left']+detection['width'],
            detection['top']+detection['height']]

def find_tracking_maximum_overlap(detections_all, video_length, missing_chance = 1, lou_max_threshold = 0.01):
    tracks_update = []
    chance_left = []
    tracks_end = []
    track_num = 0
    for i in tqdm(range(video_length)):
        frame_id = i+1
        detections_on_frame = [x for x in detections_all if x["frame"] == frame_id]
        if frame_id == 1:
            for detection in detections_on_frame:
                track_one = Track(track_num, [detection])
                track_num = track_num + 1
                tracks_update.append(track_one)
                chance_left.append(missing_chance)
        else:
            # use end flag to record the update of old tracks
            num_tracks_update = len(tracks_update)
            # remember decrease the chance
            for index in range(num_tracks_update):
                chance_left[index] -= 1
            for detection in detections_on_frame:
                lou_max = 0.0
                # here must loop end_flag instead of tracks_update, the latter will be updated
                for index in range(num_tracks_update):
                    # compare the detection and the last detection in tracks
                    box1 = detection_to_box(detection)
                    box2 = detection_to_box(tracks_update[index].detections[-1])
                    lou = bb_iou(box1, box2)
                    if lou > lou_max:
                        lou_max = lou
                        index_max = index
                if lou_max > lou_max_threshold:
                    tracks_update[index_max].detections.append(detection)
                    chance_left[index_max] = missing_chance
                else:
                    track_one = Track(track_num, [detection])
                    track_num = track_num + 1
                    tracks_update.append(track_one)
                    chance_left.append(missing_chance)
            del_index = []
            for index, chance in enumerate(chance_left):
                if chance <= 0:
                    del_index.append(index)
            for index in reversed(del_index):
                tracks_end.append(tracks_update[index])
                tracks_update.pop(index)
                chance_left.pop(index)

    tracks_end.extend(tracks_update)

    return tracks_end

