import xml.etree.ElementTree as ET
from track import Track
from tqdm import tqdm


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

