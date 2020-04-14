import numpy as np
from sort import Sort

from utils_read import transform_det


def kalman_filter_tracking_2(detections, video_n_frames, model_type):
    """
    This function assigns a track value to an object by using kalman filters.
    It also adjust bounding box coordinates based on tracking information.
    """
    bb_id_updated = []
    tracker = Sort(model_type = model_type)
    start_frame = 1
    for frame_num in range(start_frame,video_n_frames):
        #Get only bb of current frame
        dets_all_info = [x for x in detections if x['frame']==frame_num]
        dets = np.array([[bb['left'], bb['top'], bb['left']+bb['width'], bb['top']+bb['height']] for bb in dets_all_info]) #[[x1,y1,x2,y2]]
        #Apply kalman filtering
        trackers = tracker.update(dets)
        #Obtain id and bb in correct format
        for bb_dets, bb_update in zip(dets_all_info, trackers):
            bb_id_updated.append({'frame': bb_dets['frame'], 'ID': int(bb_update[4]),
                                  'left': bb_update[0], 'top': bb_update[1],
                                  'width': bb_update[2]-bb_update[0],
                                  'height': bb_update[3]-bb_update[1],
                                  'confidence': bb_dets['confidence']})
    tracks_all = transform_det(bb_id_updated)
    return tracks_all
