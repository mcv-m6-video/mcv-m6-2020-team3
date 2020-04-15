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
import PerceptualSimilarity.models as models
from PerceptualSimilarity.util import util

structure = getStructure()

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
    
video_length = len(AICityIterator(seq, cam))

currId = 0
for cam in structure[seq]:
    for track in tracks[cam]:
        if track.getIsMatched() is False:
            track.setIsMatched(True)
            track.setId(currId)
            for camComp in structure[seq]:
                if cam == camComp:
                    continue
                for trackComp in tracks[camComp]:
                    if trackComp.getIsMatched() == False:
                        #sameCar = compareTracks(cam, track, camComp, trackComp)
                        sameCar = True
                        if sameCar is True:
                            trackComp.setIsMatched(True)
                            trackComp.setId(currId)
            currId += 1

for cam in structure[seq]:
    for track in tracks[cam]:
        print(track.id)

exit()