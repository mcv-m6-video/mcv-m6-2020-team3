import utils
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

detectionFileFolder = 'Datasets/AICity/train/S03/c010/det/'
detectionFileNames = ['det_mask_rcnn.txt', 'det_ssd512.txt', 'det_yolo3.txt']
detectorNames = ['Mask RCNN', 'SSD512', 'YOLO3']

print("Reading annotations...")
groundTruth = utils.read_annotations('Datasets/AICity/aicity_annotations.xml', 2141)
groundTruthPerFrame = utils.getDetectionsPerFrame(groundTruth)
print("Computing IoU over time...")
for detectorName, detectionFile in zip(detectorNames, detectionFileNames):
    detections = utils.getDetections(detectionFileFolder + detectionFile)
    sortedDetections = utils.sortDetectionsByKey(detections, 'confidence', decreasing=True)
    detectionsPerFrame = utils.getDetectionsPerFrame(detections)
    iousPerFrame = np.zeros(len(detectionsPerFrame.keys()))
    for frame in tqdm(detectionsPerFrame.keys()):
        detectionBboxes = [utils.getBboxFromDetection(detection) for detection in detectionsPerFrame[frame]] if frame in detectionsPerFrame.keys() else []
        gtBboxes = [utils.getBboxFromDetection(gtItem) for gtItem in groundTruthPerFrame[frame]] if frame in groundTruthPerFrame.keys() else []
        results = utils.get_single_frame_results(detectionBboxes, gtBboxes, 0.5)
        if results['true_pos'] > 0:
            tmpIous = []
            for det_bbox in detectionBboxes:
                tmpIous.append(np.max([utils.bb_iou(det_bbox, gt_bbox) for gt_bbox in gtBboxes]))
                """As it is we only average the IoU if the detection is a TP, to use all detections, change the line
                    iousPerFrame[frame-1] = np.mean(finalIous)
                    for
                    iousPerFrame[frame-1] = np.mean(tmpIous)
                """
                finalIous = []
                for i, iou in enumerate(tmpIous):
                    if iou > 0.5:
                        finalIous.append(iou)
            iousPerFrame[frame-1] = np.mean(finalIous)
    plt.plot(iousPerFrame)
    plt.ylim((0, 1.0))
    plt.title('Results for ' + detectorName)
    plt.show()
#Generates a video with the bboxes for detections (blue) and gt (red)
utils.addBboxesToFrames('Datasets/AICity/frames', detections, groundTruth)
exit()
