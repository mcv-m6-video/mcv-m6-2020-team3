import motmetrics as mm


def find_frame_in_track(tracks, frame_id):
    object_id_list = []
    box_list = []
    for track_one in tracks:
        for index, detection in enumerate(track_one.detections):
            # write the rectangle
            if detection['frame'] == frame_id:
                object_id_list.append(track_one.id)
                box_list.append([detection['left'], detection['top'], detection['width'], detection['height']])
                break
    return object_id_list, box_list


def compute_idf1(groundtruth_tracks, detections_tracks, video_length):
    acc = mm.MOTAccumulator(auto_id=True)

    for i in range(video_length):
        frame_id = i + 1
        gt_ids, gt_bboxes = find_frame_in_track(groundtruth_tracks, frame_id)
        detections_ids, detections_bboxes = find_frame_in_track(detections_tracks, frame_id)

        distances_gt_det = mm.distances.iou_matrix(gt_bboxes, detections_bboxes, max_iou=1.)
        acc.update(gt_ids, detections_ids, distances_gt_det)

    print(acc.mot_events)
    mh = mm.metrics.create()
    summary = mh.compute(acc, metrics=mm.metrics.motchallenge_metrics, name='acc')
    print(summary)