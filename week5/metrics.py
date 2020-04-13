import motmetrics as mm


class MOTAccumulator(object):
    def __init__(self, auto_id=True):
        self.acc = MOTAccumulator(auto_id)

    def update(self, detections, gt):
        for frame_num, frame_detections in enumerate(detections):
            if str(frame_num) in gt.keys():
                gt_elements = gt[str(frame_num)]
                gt_bboxes = [[gt_element[0], gt_element[1], gt_element[2], gt_element[3]] for gt_element in gt_elements]
                gt_ids = [gt_element[5] for gt_element in gt_elements]
            else:
                gt_bboxes = []
                gt_ids = []

            detection_bboxes = [[detection[1], detection[2], detection[3] - detection[1],
                                 detection[4] - detection[2]] for detection in frame_detections]
            detection_ids = [detection[5] for detection in frame_detections]

            distances = distances.iou_matrix(gt_bboxes, detection_bboxes)
            self.acc.update(gt_ids, detection_ids, distances)

    def get_metrics(self):
        mh = mm.metrics.create()
        summary = mh.compute(self.acc, metrics=['num_frames', 'idf1', 'idp', 'idr'], name='acc')
        return summary