import pickle
from utils import addBboxesToFrames, calculate_mAP, bb_iou, addBboxesToFrames_gif, upscaleDetections, adjustBboxWithOpticalFlow
from utils_tracking import read_tracking_annotations, compute_mAP_track, addTracksToFrames, addTracksToFrames_gif, calculate_idf1, tracking_filter
from AICityIterator import AICityIterator, getStructure
from tqdm import tqdm
from track import Track
import opt_flow as of
import glob
import cv2
import sys
import models
from util import util
import math

import motmetrics as mm

def compute_scores(gtTracks, tracks):
    timeOffsetsForSeqs = {
        'c010': 87,
        'c011': 85,
        'c012': 59,
        'c013': 0,
        'c014': 5,
        'c015': 85
    }
    seq = 'S03'

    structure = getStructure()

    mot = mm.MOTAccumulator(auto_id=True)

    for i in tqdm(range(2516)):
        detectionIds = []
        gtIds = []
        detectionBoxes = []
        gtBoxes = []
        for cam in structure[seq]:
            for track in tracks[cam]:
                for detection in track.detections:
                    globalFrame = detection['frame']
                    if cam == 'c015':
                        continue
                        globalFrame = int(globalFrame / float(8) * 10)
                    globalFrame += timeOffsetsForSeqs[cam]
                    if globalFrame == i:
                        detectionIds.append(track.id)
                        detectionBoxes.append([detection['left'], detection['top'], detection['width'], detection['height']])
                        break
            for item in gtTracks[cam]:
                globalFrame = item['frame']
                if cam == 'c015':
                    continue
                    globalFrame = int(globalFrame / float(8) * 10)
                globalFrame += timeOffsetsForSeqs[cam]
                if globalFrame == i:
                    gtIds.append(item['ID'])
                    gtBoxes.append([item['left'], item['top'], item['width'], item['height']])
                    break
        distances = mm.distances.iou_matrix(gtBoxes, detectionBoxes)
        mot.update(gtIds, detectionIds, distances)

    mh = mm.metrics.create()
    summary = mh.compute(mot, metrics=['num_frames', 'idf1', 'idp', 'idr', 'motp', 'mota', 'precision', 'recall'], name='acc')
    print(summary)

def getPatchSimilarity(img1, img2, model):
    img0 = util.im2tensor(cv2.resize(img1, (256, 256))).cuda()
    img1 = util.im2tensor(cv2.resize(img2, (256, 256))).cuda()

    return model.forward(img0,img1)

def getBboxFromImg(img, bbox):
    box = {}
    for key in ('top', 'left', 'width', 'height'):
        box[key] = int(bbox[key])
    return  img[box['top']:box['top'] + box['height'],
            box['left']:box['left'] + box['width'],:]

def compareTracks(cam, track, camComp, trackComp, model):
    #Video offsets in frames (taken from gt)
    timeOffsetsForSeqs = {
        'c010': 87,
        'c011': 85,
        'c012': 59,
        'c013': 0,
        'c014': 5,
        'c015': 85
    }
    refImgs = AICityIterator('S03', cam).toList()
    compImgs = AICityIterator('S03', camComp).toList()
    accDst = 0.0
    matchedDetections = 0
    framesInCompTrack = [x['frame'] for x in trackComp.detections]
    for idx, detection in enumerate(track.detections):
        if idx % 1 == 0:
            frameToSearch = detection['frame']
            if cam == 'c015':
                #We "convert" the detections to 10 FPS
                frameToSearch = int(frameToSearch / float(8) * 10)
            frameToSearch += timeOffsetsForSeqs[cam]
            frameToCompare = frameToSearch - timeOffsetsForSeqs[camComp]
            if camComp == 'c015':
                #Convert frame number to 8 FPS reference
                frameToCompare = int(frameToCompare / float(10) * 8)
            if frameToCompare in framesInCompTrack:
                for det in trackComp.detections:
                    if det['frame'] == frameToCompare:
                        detectionComp = det
                        break
                matchedDetections += 1
                imgRef = cv2.imread(refImgs[detection['frame'] - 1])
                imgComp = cv2.imread(compImgs[frameToCompare - 1])

                bboxRef = getBboxFromImg(imgRef, detection)
                bboxComp = getBboxFromImg(imgComp, detectionComp)

                accDst += getPatchSimilarity(bboxRef, bboxComp, model)
    if matchedDetections == 0:
        return math.inf
    else:
        #Why
        accDst = accDst.detach().cpu().numpy()
        accDst = accDst[0][0][0][0]
        return accDst / matchedDetections

structure = getStructure()

model = models.PerceptualLoss(model='net-lin',net='alex',use_gpu=True)

seq = 'S03'

tracks = {}
gtTracks = {}
for cam in structure[seq]:
    with open("./detections/tracks_S03_{}_10.pkl".format(cam), 'rb') as p:
        tracks[cam] = pickle.load(p)
    with open("./detections/gt_tracks_S03_{}.pkl".format(cam), 'rb') as p:
        gtTracks[cam] = pickle.load(p)

for cam in structure[seq]:
    for track in tracks[cam]:
        track.setIsMatched(False)
    
if len(sys.argv) < 2:
    print("Need to specify match threshold")
    exit()
matchThr = float(sys.argv[1])

currId = 0
for cam in tqdm(structure[seq]):
    for track in tqdm(tracks[cam]):
        if track.getIsMatched() is False:
            track.setIsMatched(True)
            track.setId(currId)
            for camComp in structure[seq]:
                if cam == camComp:
                    continue
                idxBest = 0
                bestDst = math.inf
                for i, trackComp in enumerate(tracks[camComp]):
                    if trackComp.getIsMatched() == False:
                        dst = compareTracks(cam, track, camComp, trackComp, model)
                        if dst > 1 and dst < math.inf:
                            print(">1")
                        if dst < bestDst:
                            bestDst = dst
                            idxBest = i
                if bestDst < matchThr:
                    tracks[camComp][i].setIsMatched(True)
                    tracks[camComp][i].setId(currId)
            currId += 1

"""results = {}
for id in range(100):
    results[id] = 0
for cam in structure[seq]:
    for track in tracks[cam]:
        results[track.id] += 1

for id in results.keys():
    if results[id] > 0:
        print("id: {} tracks: {}".format(id, results[id]))"""

compute_scores(gtTracks, tracks)

print("-----------------End of execution-------------------")

exit()