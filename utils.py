from random import randrange, random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as pat
from xml.dom import minidom
import cv2
import glob
from tqdm import tqdm


def getDetections(detectionFilePath):
    with open(detectionFilePath, 'r') as f:
        detections = [getDictFromDetection(line) for line in f]
    return detections


def getDictFromDetection(detectionStr):
    detectionList = detectionStr.split(",")
    detectionDict = {}
    detectionDict['frame'] = int(float(detectionList[0]))
    detectionDict['left'] = int(float(detectionList[2]))
    detectionDict['top'] = int(float(detectionList[3]))
    detectionDict['width'] = int(float(detectionList[4]))
    detectionDict['height'] = int(float(detectionList[5]))
    detectionDict['confidence'] = float(detectionList[6])
    return detectionDict


def bb_iou(bboxA, bboxB):
    # This implements a function to compute the intersection over union of two bounding boxes, also known as the Jaccard Index.
    # I've adapted this code from the M1 project code we implemented. The Format of the bboxes is [tlx, tly, brx, bry, ...], where tl and br
    # indicate top-left and bottom-right corners of the bbox respectively.

    # computing coordinates of the intersection rectangle

    xA = max(bboxA[1], bboxB[1])
    yA = max(bboxA[0], bboxB[0])
    xB = min(bboxA[3], bboxB[3])
    yB = min(bboxA[2], bboxB[2])

    # compute the area of intersection rectangle
    
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both predicted and ground-truth bboxes
    
    bboxAArea = (bboxA[2] - bboxA[0] + 1) * (bboxA[3] - bboxA[1] + 1)
    bboxBArea = (bboxB[2] - bboxB[0] + 1) * (bboxB[3] - bboxB[1] + 1)

    try:
        iou = interArea / float(bboxAArea + bboxBArea - interArea)
    except ZeroDivisionError:
        iou = 0.0

    # returns the intersection over union value
    return iou



def get_noisy_bboxes(discard_probability_bbox=0.1, noise_range=20):
    """
    The function returns a dictionary with the bounding boxes for each frame where the frame number is the key. It also returns a dictionary
    that contains noisy annotations to the ground truth data. 
    """
    with open('../Datasets/AICity_data/train/S03/c010/gt/gt.txt') as f:
        lines = f.readlines()
        bboxes = dict() # stores the ground truth bboxes for each frame in the dict
        bboxes_noisy = dict() # stores the noisy annotations to the ground truth data
        num_of_instances = 0 
        for line in lines:
            num_of_instances += 1
            line = (line.split(','))
            if line[0] in bboxes.keys():
                content = [int(elem) for elem in line[1:6]]
                content.append(random())
                bboxes[line[0]].append(content)
            else:
                content = [int(elem) for elem in line[1:6]]
                content.append(random())
                bboxes[line[0]] = [content]
            if random() > discard_probability_bbox:
                if line[0] in bboxes_noisy.keys():
                    content = [int(elem) + randrange(-noise_range, noise_range) for elem in line[1:6]]
                    content.append(random())
                    bboxes_noisy[line[0]].append(content)
                else:
                    content = [int(elem) + randrange(-noise_range, noise_range) for elem in line[1:6]]
                    content.append(random())
                    bboxes_noisy[line[0]] = [content]

    return bboxes, bboxes_noisy, num_of_instances


def read_xml_annotations(annotations_path):
    '''The function reads the xml files and returns the corresponding bboxes for every frame.'''
    files = os.listdir(annotations_path)

    bboxes = dict()
    for file_ in files:
        xmldoc = minidom.parse(annotations_path + file_)
        bboxes_list = xmldoc.getElementsByTagName('box')
        for element in bboxes_list:
            frame = element.getAttribute('frame')
            tlx = int(float(element.getAttribute('tlx')))
            tly = int(float(element.getAttribute('tly')))
            brx = int(float(element.getAttribute('brx')))
            bry = int(float(element.getAttribute('bry')))
            width = ybr - ytl
            height = xbr - xtl

            if frame in bboxes.keys():
                bboxes[frame].append([-1, xtl, ytl, height, width, random()])
            else:
                bboxes[frame] = [[-1, xtl, ytl, height, width, random()]]
    return bboxes


def sortDetectionsByKey(detectionList, key, decreasing=False):
    sortedList = sorted(detectionList, key=lambda k: k[key], reverse=decreasing)
    return sortedList


def getBboxFromDetection(detection):
    bbox = np.zeros(4)
    bbox[0] = detection['left']
    bbox[1] = detection['top']
    bbox[2] = detection['left'] + detection['width']
    bbox[3] = detection['left'] + detection['height']
    return bbox

def addBboxesToFrames(framesPath, detections, groundTruth):
    #Show GT bboxes and detections
    #Preprocess detections and GT
    for detection in detections:
        detection['isGT'] = False
    for item in groundTruth:
        item['isGT'] = True

    combinedList = detections + groundTruth

    combinedList = sortDetectionsByKey(combinedList, 'frame')

    frameFiles = glob.glob(framesPath + '/*.jpg')
    size = (1920, 1080)
    fps = 10
    out = cv2.VideoWriter('bboxes.avi',cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
    
    prevFrame = 0
    frameMat = cv2.imread(frameFiles[0])
    for item in tqdm(combinedList[0:5000]):
        frame = item['frame'] - 1
        if frame != prevFrame:
            out.write(frameMat)
            frameMat = cv2.imread(frameFiles[frame])
        startPoint = (item['left'], item['top'])
        endPoint = (startPoint[0] + item['width'], startPoint[1] + item['height'])
        color = (255, 0, 0) if item['isGT'] is True else (0, 0, 255)
        frameMat = cv2.rectangle(frameMat, startPoint, endPoint, color, 2)
        prevFrame = frame
    out.release()

    