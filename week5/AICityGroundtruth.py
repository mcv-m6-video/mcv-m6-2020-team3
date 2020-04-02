import os
import glob
from utils import getDictFromDetection

def getGroundtruth(seq, cam, frame):
    datasetStructure = {
        'S01': ['c001', 'c002', 'c003', 'c004', 'c005'],
        'S03': ['c010', 'c011', 'c012', 'c013', 'c014', 'c015'],
        'S04': ['c016', 'c018', 'c020', 'c022', 'c024', 'c026', 'c028', 'c030', 'c032', 'c034', 'c036', 'c038', 'c040','c017', 'c019', 'c021', 'c023', 'c025', 'c027', 'c029', 'c031', 'c033', 'c035', 'c037', 'c039']
    }

    if os.path.exists('Datasets/AIC20_track3/train/'):
        baseDir = 'Datasets/AIC20_track3/train/'
    elif os.path.exists('../Datasets/AIC20_track3/train/'):
        baseDir = '../Datasets/AIC20_track3/train/'
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
            if currFrame == frame:
                detections.append(getDictFromDetection(line, True))
            if currFrame > frame:
                break

    return detections

