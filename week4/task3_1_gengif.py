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

def box_to_detection(box):
    detection = {}
    detection['left'] = box[0]
    detection['top'] = box[1]
    detection['width'] = box[2] - box[0]
    detection['height'] = box[3] - box[1]
    return detection

if __name__ == "__main__":
    detections_filename = "./detections/results_t12_interleaved.pkl"
    crop_center = True
    
    #video_length = 2141
    video_length = 300
    video_path = "./Datasets/AICity/frames/"
    frameFiles = glob.glob(video_path + '/*.jpg')
    frameFiles = sorted(frameFiles)

    print("Reading pkl")
    with open(detections_filename, 'rb') as p:
        detections = pickle.load(p)
        p.close()

    detections = upscaleDetections(detections)
    adjusted_detections = []

    curr_frame = cv2.imread(frameFiles[0])
    prev_frame = cv2.imread(frameFiles[1])
    for i in tqdm(range(video_length)):
        detections_on_frame = [x for x in detections if x["frame"] == i+1]
        if i == 0:            
            for detection in detections_on_frame:
                adjusted_detections.append(detection)
        else:
            detections_on_frame = [x for x in detections if x["frame"] == i]
            prev_frame = curr_frame
            curr_frame = cv2.imread(frameFiles[i])
            #frameOF = of.farneback(prev_frame, curr_frame)
            for detection in detections_on_frame:
                box = detection_to_box(detection)
                """adjusted_box = adjustBboxWithOpticalFlow(box, frameOF, crop_center)
                adjusted_box = box_to_detection(adjusted_box)"""
                adjusted_box = detection
                tmp_detection = {}
                tmp_detection['frame'] = i+1
                tmp_detection['left'] = adjusted_box['left']
                tmp_detection['top'] = adjusted_box['top']
                tmp_detection['width'] = adjusted_box['width']
                tmp_detection['height'] = adjusted_box['height']
                tmp_detection['confidence'] = detection['confidence']
                adjusted_detections.append(tmp_detection)

    addBboxesToFrames_gif(video_path, adjusted_detections, detections, start_frame=210, end_frame=260, name="test")

            


    