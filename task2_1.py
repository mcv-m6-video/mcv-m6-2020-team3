"""
example
{'frame': 100, 'left': 931, 'top': 78, 'width': 82, 'height': 68, 'confidence': 0.99}
Note, frame starts from 1.
"""
import pickle
from utils_w3 import addBboxesToFrames, read_annotations
from utils_tracking import read_tracking_annotations

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
    # groundTruth = read_annotations(groundtruth_xml_path, video_length)
    groundTruth, tracks_gt_list = read_tracking_annotations(groundtruth_xml_path, video_length)
    addBboxesToFrames('Datasets/AICity/frames', detections, groundTruth, "test")

