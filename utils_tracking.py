import xml.etree.ElementTree as ET
from track import Track
from tqdm import tqdm
import numpy as np
from utils_w3 import calculate_mAP
import glob
import cv2
import imageio


def detection_is_dictionary(frame_id, left, top, width, height, confidence = 1.0):
    detection = {}
    detection['frame'] = frame_id
    detection['left'] = left
    detection['top'] = top
    detection['width'] = width
    detection['height'] = height
    detection['confidence'] = confidence
    return detection

def read_tracking_annotations(annotation_path, video_length):
    """
    Arguments:
    capture: frames from video, opened as cv2.VideoCapture
    root: parsed xml annotations as ET.parse(annotation_path).getroot()
    """
    root = ET.parse(annotation_path).getroot()

    ground_truths = []
    tracks = []
    images = []
    num = 0

    for num in tqdm(range(video_length)):
        #for now: (take only numannotated annotated frames)
        #if num > numannotated:
        #    break

        for track in root.findall('track'):
            gt_id = track.attrib['id']
            label = track.attrib['label']
            box = track.find("box[@frame='{0}']".format(str(num)))

            #if box is not None and (label == 'car' or label == 'bike'):    # Read cars and bikes
            if box is not None and label == 'car':                          # Read cars

                if box.attrib['occluded'] == '1':                           # Discard occluded
                    continue

                #if label == 'car' and box[0].text == 'true':               # Discard parked cars
                #    continue

                frame = int(box.attrib['frame'])
                #if frame < 534:
                #    continue

                xtl = int(float(box.attrib['xtl']))
                ytl = int(float(box.attrib['ytl']))
                xbr = int(float(box.attrib['xbr']))
                ybr = int(float(box.attrib['ybr']))
                ground_truths.append(detection_is_dictionary(frame, xtl, ytl, xbr - xtl + 1, ybr - ytl + 1, 1))
                track_corresponding = [t for t in tracks if t.id == gt_id]
                if len(track_corresponding) > 0:
                    track_corresponding[0].detections.append(detection_is_dictionary(frame+1, xtl, ytl, xbr - xtl + 1, ybr - ytl + 1, 1))
                else:
                    track_corresponding = Track(gt_id, [detection_is_dictionary(frame+1, xtl, ytl, xbr - xtl + 1, ybr - ytl + 1, 1)])
                    tracks.append(track_corresponding)

    # print(ground_truths)
    return ground_truths, tracks


def compute_mAP_track(groundtruth_tracks, detections_tracks, IoU_threshold=0.5, verbose = False):
    """
    Calculate the mean of the best mAP for each track of the ground truth
    input:
    groundtruth_tracks: tracks of gt, the format should be list of Class Track
    detections_tracks: tracks of detections, the format should be list of Class Track
    """

    mAP_list = list()

    for groundtruth_track in tqdm(groundtruth_tracks):
        #print('DETECTION'), print(detection_track)
        max_mAP = 0

        for detection_track in detections_tracks:
            #print(groundtruth_track)
            mAP = calculate_mAP(groundtruth_track.detections, detection_track.detections,
                                          IoU_threshold)
            #print(mAP)
            if(max_mAP < mAP):
                max_mAP = mAP

        mAP_list.append(max_mAP)

    np_mAP_list = np.asarray(mAP_list)

    mAP_mean = np.mean(np_mAP_list)
    if verbose:
        print("np_mAP_list:")
        print(np_mAP_list)
        print("mAP = {}".format(mAP_mean))

    return mAP_mean


def center_of_detection(detection):
    return (int(detection['left'] + 0.5 * detection['width']), int(detection['top'] + 0.5 * detection['height']))


def write_one_frame(detections_tracks, frame_id, frameMat, color):
    """
    tool for addTracksToFrames
    :param detections_tracks: this can be detections or ground truth
    :param frame_id: which frame
    :param frameMat: frame picture
    :param color: rectangle and line color
    :return: frameMat
    """
    for track_one in detections_tracks:
        index = 0
        flag_shoot = False
        for index, detection in enumerate(track_one.detections):
            # write the rectangle
            if detection['frame'] == frame_id:
                startPoint = (int(detection['left']), int(detection['top']))
                endPoint = (int(startPoint[0] + detection['width']), int(startPoint[1] + detection['height']))
                frameMat = cv2.rectangle(frameMat, startPoint, endPoint, color, 2)
                flag_shoot = True
                break
        if flag_shoot:
            shoot_index = index
            # write the line
            for index in range(shoot_index - 1):
                startPoint = center_of_detection(track_one.detections[index])
                endPoint = center_of_detection(track_one.detections[index + 1])
                frameMat = cv2.line(frameMat, startPoint, endPoint, color, 2)
    return frameMat


def addTracksToFrames(framesPath, detections_tracks, tracks_gt_list, start_frame = 1, end_frame = 2141, name = "test"):
    """
    write the video of the tracking result
    :param framesPath: path of frames
    :param detections_tracks: detections in list of Track. Detections in Track should be sorted.
    :param tracks_gt_list: ground truth in list of Track. Detections in Track should be sorted.
    :param name: name of video.
    :return: None
    """
    size = (1920, 1080)
    fps = 10
    out = cv2.VideoWriter(name + '.avi', cv2.VideoWriter_fourcc(*'DIVX'), fps, size)

    for frame_id in tqdm(range(start_frame, end_frame)):
        filename = "{}{}.jpg".format(framesPath, str(frame_id).zfill(5))
        frameMat = cv2.imread(filename)
        color_detection = (0, 0, 255)
        write_one_frame(detections_tracks, frame_id, frameMat, color_detection)
        color_gt = (255, 0, 0)
        write_one_frame(tracks_gt_list, frame_id, frameMat, color_gt)
        out.write(frameMat)
    out.release()


def addTracksToFrames_gif(framesPath, detections_tracks, tracks_gt_list, start_frame = 1, end_frame = 10, name = "test"):
    """
    write the gif of the tracking result
    :param framesPath: path of frames
    :param detections_tracks: detections in list of Track. Detections in Track should be sorted.
    :param tracks_gt_list: ground truth in list of Track. Detections in Track should be sorted.
    :param name: name of video.
    :return: None
    """
    images = []

    for frame_id in tqdm(range(start_frame, end_frame)):
        filename = "{}{}.jpg".format(framesPath, str(frame_id).zfill(5))
        frameMat = cv2.imread(filename)
        color_detection = (0, 0, 255)
        write_one_frame(detections_tracks, frame_id, frameMat, color_detection)
        color_gt = (255, 0, 0)
        write_one_frame(tracks_gt_list, frame_id, frameMat, color_gt)
        resized = cv2.resize(frameMat, (480, 270), interpolation=cv2.INTER_AREA)
        images.append(resized)
    imageio.mimsave(name + '.gif', images)

