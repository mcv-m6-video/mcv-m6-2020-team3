from __future__ import print_function
from sort import Sort
from utils_tracking import read_tracking_annotations, compute_mAP_track
import pickle
import numpy as np
import cv2

def main(visualize = False):
    detections_filename = "./detections/detections.pkl"
    video_length = 2141
    video_path = "./Datasets/AICity/frames/"
    groundtruth_xml_path = 'Datasets/AICity/aicity_annotations.xml'

    print("Reading pkl")
    with open(detections_filename, 'rb') as p:
        detections = pickle.load(p)
        p.close()
    new = []
    for detect in detections:
        new.append(np.fromiter(detect.values(), dtype=float))
    new1 = np.asarray(new)
    kalman_tracker = Sort()
    trackers = kalman_tracker.update(new1)

    print("Reading annotations...")
    groundTruth, tracks_gt_list = read_tracking_annotations(groundtruth_xml_path, video_length)

    whole_video_detections = []
    current_frame_detections = []

    for track_det in trackers:
        track_det = track_det.astype(np.uint32)
        current_frame_detections.append(['car', track_det[0], track_det[1], track_det[2],
                                         track_det[3], track_det[4]])

        if visualize is True:
            cv2.rectangle(frame, (track_det[1], track_det[0]), (track_det[3], track_det[2]), (0, 0, 255), 3)

            font = cv2.FONT_HERSHEY_DUPLEX
            placement = (track_det[3] + 10, track_det[2] + 10)
            font_scale = 1
            font_color = (0, 255, 0)
            line_type = 2

            cv2.putText(frame, str(track_det[4]), placement, font, font_scale, font_color, line_type)

    whole_video_detections.append(current_frame_detections)

    if visualize is True:
        cv2.imwrite('tracking_videos/kalman/image%04i.jpg' % frame_idx, frame)
        cv2.imshow('output', frame)
        cv2.waitKey(10)

    mAP_track = compute_mAP_track(tracks_gt_list, whole_video_detections, IoU_threshold=0.5)
    print("mAP_track = ", mAP_track)


if __name__ == "__main__":
    main(visualize=False)






