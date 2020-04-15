import cv2
import os
import glob
import re

class AICityIterator:
    datasetStructure = {
        'S01': ['c001', 'c002', 'c003', 'c004', 'c005'],
        'S03': ['c010', 'c011', 'c012', 'c013', 'c014', 'c015'],
        'S04': ['c016', 'c018', 'c020', 'c022', 'c024', 'c026', 'c028',
            'c030', 'c032', 'c034', 'c036', 'c038', 'c040','c017',
            'c019', 'c021', 'c023', 'c025', 'c027', 'c029', 'c031', 'c033', 'c035', 'c037', 'c039']
    }

    def __init__(self, seq, camera, videoLength=None):
        if os.path.exists('Datasets/AIC20_track3/train/'):
            self.baseDir = 'Datasets/AIC20_track3/train/'
        elif os.path.exists('../../Datasets/AIC20_track3/train/'):
            self.baseDir = '../../Datasets/AIC20_track3/train/'
        else:
            raise Exception("Dataset folder not found")

        if seq not in self.datasetStructure.keys():
            raise Exception('Invalid sequence: ' + seq)

        if camera not in self.datasetStructure[seq]:
            raise Exception('Invalid camera ' + camera + ' for sequence ' + seq)

        datasetDir = self.baseDir + seq + '/' + camera + '/frames/'
        self.frameFiles = glob.glob(datasetDir + '*.jpg')
        self.frameFiles = sorted(self.frameFiles)
        self.limit = videoLength if videoLength is not None else len(self.frameFiles)
        self.currItem = -1

    def __iter__(self):
        return self

    def __next__(self):
        self.currItem += 1
        if self.currItem < self.limit:
            return self.frameFiles[self.currItem]
        raise StopIteration


def getDictFromDetection(detectionStr, isGT=False):
    detectionList = detectionStr.split(",")
    detectionDict = {}
    detectionDict['frame'] = int(float(detectionList[0]))
    detectionDict['ID'] = int(detectionList[1])
    detectionDict['left'] = int(float(detectionList[2]))
    detectionDict['top'] = int(float(detectionList[3]))
    detectionDict['width'] = int(float(detectionList[4]))
    detectionDict['height'] = int(float(detectionList[5]))
    if isGT is not True:
        detectionDict['confidence'] = float(detectionList[6])
    return detectionDict


def get_gt_txt(seq, cam):
    datasetStructure = {
        'S01': ['c001', 'c002', 'c003', 'c004', 'c005'],
        'S03': ['c010', 'c011', 'c012', 'c013', 'c014', 'c015'],
        'S04': ['c016', 'c018', 'c020', 'c022', 'c024', 'c026', 'c028', 'c030', 'c032', 'c034', 'c036', 'c038', 'c040',
                'c017', 'c019', 'c021', 'c023', 'c025', 'c027', 'c029', 'c031', 'c033', 'c035', 'c037', 'c039']
    }

    if os.path.exists('Datasets/AIC20_track3/train/'):
        baseDir = 'Datasets/AIC20_track3/train/'
    elif os.path.exists('../../Datasets/AIC20_track3/train/'):
        baseDir = '../../Datasets/AIC20_track3/train/'
    else:
        raise Exception("Dataset folder not found")

    if seq not in datasetStructure.keys():
        raise Exception('Invalid sequence: ' + seq)

    if cam not in datasetStructure[seq]:
        raise Exception('Invalid camera ' + cam + ' for sequence ' + seq)

    gtFilePath = baseDir + seq + '/' + cam + '/gt/gt.txt'

    detections = []
    with open(gtFilePath, 'r') as f:
        for line in f:
            splitStr = line.split(',')
            currFrame = int(splitStr[0])
            detections.append(getDictFromDetection(line, True))

    det_frames = []
    last_frame = 0
    det_one_frame = []
    for detection in detections:
        if detection['frame'] != last_frame:
            for current_frame in range(last_frame, detection['frame']):
                if current_frame != 0:
                    det_frames.append(det_one_frame)
                det_one_frame = []
        det_one_frame.append(detection)
        last_frame = detection['frame']

    det_frames.append(det_one_frame)

    return det_frames


def load_from_dataset(datasetStructure, dataset):
    dataset_dicts = []
    idx = 0
    for seq in dataset:
        cam_list = datasetStructure[seq]
        for cam in cam_list:
            # print (seq, cam)
            gt_txt = get_gt_txt(seq, cam)
            new_cam = True
            for imgPath in AICityIterator(seq, cam):
                # print(imgPath)
                frame = int(imgPath.split('\\')[-1].split(".")[0])
                # linux use /, windows use \\
                # frame = int(imgPath.split('/')[-1].split(".")[0])

                if (frame - 1) < len(gt_txt):
                    v = gt_txt[frame - 1]
                else:
                    v = []
                # print(v)

                record = {}

                record["file_name"] = imgPath
                record["image_id"] = idx
                idx = idx + 1

                objs = []
                for instance in v:
                    box = [instance['left'], instance['top'], instance['width'],
                           instance['height']]
                    obj = {
                        "ID": instance['ID'],
                        "box": box,
                    }
                    objs.append(obj)
                record["objs"] = objs

                dataset_dicts.append(record)
    return dataset_dicts

