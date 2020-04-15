import pickle
from distance_metric_learning import train_metric


if __name__ == "__main__":
    timestamp = {
        'c010': 8.715,
        'c011': 8.457,
        'c012': 5.879,
        'c013': 0,
        'c014': 5.042,
        'c015': 8.492,
    }


    # read the result of tracing with single camera.
    with open("detections_tracks_all_camera.pkl", 'rb') as f:
        detections_tracks_all_camera = pickle.load(f)
        f.close()

    for cam in detections_tracks_all_camera.keys():
        for track_one in detections_tracks_all_camera[cam]:
            for detection in track_one.detections:
                detection['cam'] = cam

    nca = train_metric()
