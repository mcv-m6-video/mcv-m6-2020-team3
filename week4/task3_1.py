"""
example
{'frame': 100, 'left': 931, 'top': 78, 'width': 82, 'height': 68, 'confidence': 0.99}
Note, frame starts from 1.
"""
import pickle
from utils_w4 import addBboxesToFrames, calculate_mAP, bb_iou, addBboxesToFrames_gif, upscaleDetections, adjustBboxWithOpticalFlow
from utils_tracking import read_tracking_annotations, compute_mAP_track, addTracksToFrames, addTracksToFrames_gif
from tqdm import tqdm
from track import Track
import opt_flow as of
import glob
import cv2
import sys

def detection_to_box(detection):
    return [detection['left'], detection['top'], detection['left']+detection['width'],
            detection['top']+detection['height']]

def find_tracking(detections_all, video_length, missing_chance = 1, lou_max_threshold = 0.01, use_of=False, crop_center=False):
    video_path = "./Datasets/AICity/frames/"
    frameFiles = glob.glob(video_path + '/*.jpg')
    frameFiles = sorted(frameFiles)

    tracks_update = []
    chance_left = []
    tracks_end = []
    track_num = 0
    curr_frame = cv2.imread(frameFiles[0])
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
            prev_frame = curr_frame
            curr_frame = cv2.imread(frameFiles[frame_id-1])
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
    #detections_filename = "./detections/results_t12_first.pkl"
    detections_filename = "./detections/results_t12_interleaved.pkl"
    use_of = False if len(sys.argv) > 1 and sys.argv[1] == "-n" else True
    crop_center = True if len(sys.argv) > 1 and sys.argv[1] == "-c" else False
    
    video_length = 2141
    #video_length = 500
    video_path = "./Datasets/AICity/frames/"
    groundtruth_xml_path = 'Datasets/AICity/aicity_annotations.xml'

    print("Reading pkl")
    with open(detections_filename, 'rb') as p:
        detections = pickle.load(p)
        p.close()

    detections = upscaleDetections(detections)

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

    #addBboxesToFrames_gif(video_path, detections, groundTruth, start_frame=210, end_frame=260, name="test")

    # sort detections because detections is sort by confidence while calculating map.
    detections.sort(key=lambda x: x['frame'])

    # detections_tracks = find_tracking(detections, video_length, missing_chance=1, lou_max_threshold=0.01)
    missing_chance_list = [1, 3, 5, 7]
    lou_max_threshold_list = [0.01, 0.1, 0.3, 0.5, 0.7]
    missing_chance_list = [3, 5, 7]
    lou_max_threshold_list = [0.3, 0.5, 0.7]
    result_list = []
    for missing_chance in missing_chance_list:
        for lou_max_threshold in lou_max_threshold_list:
            detections_tracks = find_tracking(detections, video_length, missing_chance=missing_chance,
                                              lou_max_threshold=lou_max_threshold, use_of=use_of, crop_center=crop_center)
            with open("track_gt_de.pkl", 'wb') as f:
                pickle.dump([tracks_gt_list, detections_tracks, video_length], f)
                f.close()
            mAP_track = compute_mAP_track(tracks_gt_list, detections_tracks, IoU_threshold=0.5)
            print("Missing chance: " + str(missing_chance))
            print("IoU max thr: " + str(lou_max_threshold))
            print("mAP_track = ", mAP_track)
            result_list.append([missing_chance, lou_max_threshold, mAP_track])
    print(result_list)

    for track_one in detections_tracks:
        track_one.detections.sort(key=lambda x: x['frame'])

    #Plot next frame and previous frame adjusted with of
    exit()

    addTracksToFrames_gif(video_path, detections_tracks, tracks_gt_list, start_frame=210, end_frame=390, name="test")




