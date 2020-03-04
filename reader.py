import cv2
import xml.etree.ElementTree as ET

from tqdm import tqdm
import Detection


def read_detections(path: str):
    # [frame, -1, left, top, width, height, conf, -1, -1, -1]
    frame_detections = []

    with open(path) as f:
        for line in f.readlines():
            parts = line.split(',')
            frame_id = int(parts[0])
            # while frame_id > len(frame_detections):
            #     frame_detections.append([])

            tl_x = int(float(parts[2]))
            tl_y = int(float(parts[3]))
            width = int(float(parts[4]))
            height = int(float(parts[5]))

            frame_detections.append(Detection(frame_id, 'car', tl_x, tl_y, width, height, 1))

    return frame_detections


def read_annotations(annotation_path, video_path):
    """
    Arguments: 
    capture: frames from video, opened as cv2.VideoCapture
    root: parsed xml annotations as ET.parse(annotation_path).getroot()
    """
    capture = cv2.VideoCapture(video_path)
    root = ET.parse(annotation_path).getroot()

    ground_truths = []
    images = []
    num = 0

    pbar = tqdm(total=2140)

    while capture.isOpened():
        valid, image = capture.read()
        if not valid:
            break
        #for now: (take only numannotated annotated frames)
        #if num > numannotated:
        #    break

        images.append(image)
        for track in root.findall('track'):
            #gt_id = track.attrib['id']
            label = track.attrib['label']
            box = track.find("box[@frame='{0}']".format(str(num)))
            if box is not None and (label == 'car' or label == 'bike'):

                if box.attrib['occluded'] == '1':
                    continue

                if label == 'car' and box[0].text == 'true':                # Check parked
                    continue

                frame = int(box.attrib['frame'])
                xtl = int(float(box.attrib['xtl']))
                ytl = int(float(box.attrib['ytl']))
                xbr = int(float(box.attrib['xbr']))
                ybr = int(float(box.attrib['ybr']))
                ground_truths.append(Detection(frame, label, xtl, ytl, xbr - xtl + 1, ybr - ytl + 1, 1))
                #ground_truths.append(Detection(frame, label, xtl, ytl, xbr, ybr, 1))

        pbar.update(1)
        num += 1

    pbar.close()

    # print(ground_truths)
    capture.release()
    return ground_truths


def read_annotations_from_txt(gt_path, analyze=True):
    """
    Read annotations from the txt files
    Arguments:
    gt_path: path to .txt file
    :returns: list of Detection
    """
    ground_truths_list = list()
    if analyze:
        max_w = 0
        min_w = 2000
        max_h = 0
        min_h = 2000
        min_ratio = 100
        max_ratio = 0
    with open(gt_path) as f:
        for line in f:
            data = line.split(',')
            ground_truths_list.append(Detection(int(data[0]), 'car', int(float(data[2])), int(float(data[3])), int(float(data[4])), int(float(data[5])),float(data[6])))

            if analyze:
                if int(data[4]) < min_w: min_w = int(data[4])
                if int(data[4]) > max_w: max_w = int(data[4])
                if int(data[5]) < min_h: min_h = int(data[5])
                if int(data[5]) > max_h: max_h = int(data[5])
                if int(data[5])/int(data[4]) > max_ratio: max_ratio = int(data[5])/int(data[4])
                if int(data[5])/int(data[4]) < min_ratio: min_ratio = int(data[5])/int(data[4])
    # print('width: [{0}, {1}]'.format(min_w, max_w))
    # print('height: [{0}, {1}]'.format(min_h, max_h))
    # print('ratio: [{0}, {1}]'.format(min_ratio, max_ratio))

    return ground_truths_list


def read_annotations_file(gt_path, video_path):
    if (gt_path.endswith('.txt')):
        annotations_list = read_annotations_from_txt(gt_path)
    elif (gt_path.endswith('.xml')):
        annotations_list = read_annotations(gt_path, video_path)
    else:
        raise Exception('Incompatible filetype')

    return annotations_list