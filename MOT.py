import numpy as np
import motmetrics as mm
from sklearn.metrics.pairwise import pairwise_distances


class MOT:
    
    def __init__(self):
        self.acc = mm.MOTAccumulator(auto_id=True)

    def update(self, detections, gts):
        detection_on_frame = [x for x in detections_list if x['frame'] == groundtruth[0]['frame']]
        detection_bboxes = [box(o) for o in detection_on_frame]
        gt_on_frame = [x for x in groundtruth_list if x['frame'] == detection['frame']]
        gt_bboxes = [o.bbox for o in gt_on_frame]
        
        distances_matrix = mm.distances.iou_matrix(gt_bboxes, detection_bboxes, max_iou=1.)

    distances_matrix = mm.distances.iou_matrix(gt_bboxes, detection_bboxes, max_iou=1.)

        self.acc.update([det.id for det in detection_on_frame], [gt.id for gt in gt_on_frame], distances_matrix)

    def get_idf1(self):
        mh = mm.metrics.create()
        idf1 = mh.compute(self.acc, metrics=['idf1'], name='acc')
        return idf1
    
