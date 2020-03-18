"""
example
{'frame': 100, 'left': 931, 'top': 78, 'width': 82, 'height': 68, 'confidence': 0.99}
Note, frame starts from 1.
"""
import pickle
from utils_w3 import addBboxesToFrames, calculate_mAP, bb_iou
from utils_tracking import read_tracking_annotations, compute_mAP_track, addTracksToFrames, addTracksToFrames_gif
from tqdm import tqdm
from track import Track

def detection_to_box(detection):
    return [detection['left'], detection['top'], detection['left']+detection['width'],
            detection['top']+detection['height']]

def find_tracking(detections_all, video_length, missing_chance = 5):
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
                if lou_max > 0.01:
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

if __name__ == "__main__":
    detections_filename = "./detections/detections.pkl"
    video_length = 2141
    video_path = "./Datasets/AICity/frames/"
    groundtruth_xml_path = 'Datasets/AICity/aicity_annotations.xml'

    print("Reading pkl")
    with open(detections_filename, 'rb') as p:
        detections = pickle.load(p)
        p.close()

    print("Reading annotations...")
    read_annotations_flag = False
    annotations_pkl_filename = "gt_annotations.pkl"
    if read_annotations_flag:
        # groundTruth = read_annotations(groundtruth_xml_path, video_length)
        groundTruth, tracks_gt_list = read_tracking_annotations(groundtruth_xml_path, video_length)
        with open(annotations_pkl_filename, 'wb') as f:
            pickle.dump([groundTruth, tracks_gt_list], f)
            f.close()
    else:
        print("Reading pkl")
        with open(annotations_pkl_filename, 'rb') as p:
            groundTruth, tracks_gt_list = pickle.load(p)
            p.close()

    # print("calculate mAP...")
    # mAP = calculate_mAP(groundTruth, detections, IoU_threshold=0.5, have_confidence=True, verbose=True)
    # print("mAP = ", mAP)

    # addBboxesToFrames('Datasets/AICity/frames', detections, groundTruth, "test")

    detections_tracks = find_tracking(detections, video_length, missing_chance=1)
    mAP_track = compute_mAP_track(tracks_gt_list, detections_tracks, IoU_threshold=0.5)
    print("mAP_track = ", mAP_track)

    # addTracksToFrames(video_path, detections_tracks, tracks_gt_list, start_frame=1, end_frame=1000, name="test")
    addTracksToFrames_gif(video_path, detections_tracks, tracks_gt_list, start_frame=800, end_frame=930, name="test")




