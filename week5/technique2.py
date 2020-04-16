"""
example
{'frame': 100, 'left': 931, 'top': 78, 'width': 82, 'height': 68, 'confidence': 0.99}
Note, frame starts from 1.
"""

import pickle
from utils import addBboxesToFrames, calculate_mAP, bb_iou, addBboxesToFrames_gif, upscaleDetections, adjustBboxWithOpticalFlow
from utils_tracking import read_tracking_annotations, compute_mAP_track, addTracksToFrames, addTracksToFrames_gif, calculate_idf1, tracking_filter
from AICityIterator import AICityIterator, getStructure
from tqdm import tqdm
from track import Track
import opt_flow as of
import glob
import cv2
import sys

def detection_to_box(detection):
    return [detection['left'], detection['top'], detection['left']+detection['width'],
            detection['top']+detection['height']]

def find_tracking(seq, cam, detections_all, video_length, missing_chance = 1, lou_max_threshold = 0.01, use_of=False, crop_center=False):

    tracks_update = []
    chance_left = []
    tracks_end = []
    track_num = 0
    iterator = AICityIterator(seq, cam, video_length)

    for i, imgPath in tqdm(enumerate(iterator), total=len(iterator)):
        frame_id = i+1
        detections_on_frame = [x for x in detections_all if x["frame"] == frame_id]
        if frame_id == 1:
            curr_frame = cv2.imread(imgPath)
            for detection in detections_on_frame:
                track_one = Track(track_num, [detection])
                track_num = track_num + 1
                tracks_update.append(track_one)
                chance_left.append(missing_chance)
        else:
            prev_frame = curr_frame
            curr_frame = cv2.imread(imgPath)
            if use_of is True:
                frameOF = of.farneback(prev_frame, curr_frame)
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
                    if use_of is True:
                        adjusted_box2 = adjustBboxWithOpticalFlow(box2, frameOF, crop_center)
                    else:
                        adjusted_box2 = box2
                    lou = bb_iou(box1, adjusted_box2)
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

if __name__ == "__main__":
    use_of = False if "-n" in sys.argv else True
    crop_center = True if "-c" in sys.argv else False
    filterStatic = True if "-f" in sys.argv else False
    isFasterRCNN = "_frcnn" if "-frcnn" in sys.argv else ""

    seq = sys.argv[1] if len(sys.argv) >= 3 else 'S03'
    cam = sys.argv[2] if len(sys.argv) >= 3 else 'c010'

    detections_filename = "./detections/detections{}_{}_{}.pkl".format(isFasterRCNN, seq, cam)
    
    video_length = len(AICityIterator(seq, cam))

    print("Reading pkl")
    with open(detections_filename, 'rb') as p:
        detections = pickle.load(p)
        p.close()

    print("Reading annotations...")
    groundTruth, tracks_gt_list = read_tracking_annotations(seq, cam, video_length)

    # sort detections because detections is sort by confidence while calculating map.
    detections.sort(key=lambda x: x['frame'])

    baseDatasetPath = 'Datasets/AIC20_track3/train/'
    #Best parameters
    frameDistanceList = [10, 5] if filterStatic else ["noFilter"]
    trackLenList = [100, 1000] if filterStatic else ["noFilter"]
    missing_chance_list = [3]
    lou_max_threshold_list = [0.3]
    for missing_chance in missing_chance_list:
        for lou_max_threshold in lou_max_threshold_list:
            detections_tracks = find_tracking(seq, cam, detections, video_length, missing_chance=missing_chance,
                                              lou_max_threshold=lou_max_threshold, use_of=use_of, crop_center=crop_center)
            if filterStatic is True:
                orig_tracks = detections_tracks
            for frameDistance, trackLen in zip(frameDistanceList, trackLenList):
                if filterStatic is True:
                    print("Length of detections pre-filter: ", len(orig_tracks))
                    detections_tracks = tracking_filter(orig_tracks, frameDistance, trackLen)
                mAP_track = compute_mAP_track(tracks_gt_list, detections_tracks, IoU_threshold=0.5)
                print("Missing chance: " + str(missing_chance))
                print("IoU max thr: " + str(lou_max_threshold))
                print("mAP_track = ", mAP_track)
                if filterStatic is True:
                    print("Frame diff: ", frameDistance)
                    print("Minimum track duration: ", trackLen)
                for track_one in detections_tracks:
                    track_one.detections.sort(key=lambda x: x['frame'])

                print("Length of detections post-filter: ", len(detections_tracks))

                calculate_idf1(groundTruth, detections_tracks, video_length)
                #addTracksToFrames(baseDatasetPath + '{}/{}/frames/'.format(seq, cam), detections_tracks, tracks_gt_list,
                        #start_frame=1, end_frame=video_length, name="test_track_{}_{}_{}".format(seq, cam, frameDistance))
                with open('detections/tracks_{}_{}_{}.pkl'.format(seq, cam, frameDistance), "wb") as f:
                    pickle.dump(detections_tracks, f)



