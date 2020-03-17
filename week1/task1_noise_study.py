import utils
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

detectionFileFolder = '../Datasets/AICity/train/S03/c010/det/'
detectionFileNames = ['det_mask_rcnn.txt', 'det_ssd512.txt', 'det_yolo3.txt']
detectorNames = ['Mask RCNN', 'SSD512', 'YOLO3']

#too long will need a lot of time, so decrease
video_length = 100

print("Reading annotations...")
groundTruth = utils.read_annotations('../Datasets/AICity/aicity_annotations.xml', video_length)
groundTruthPerFrame = utils.getDetectionsPerFrame(groundTruth)

verbose = False
IoU_mean_list = []
mAP_mean_list = []
noise_range_list = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
for noise_range in noise_range_list:
    print("noise range = ", noise_range)
    # bboxes, bboxes_noisy, num_of_instances = utils.get_noisy_bboxes(discard_probability_bbox=0.1, noise_range=20)
    # noisy_gt_boxes = utils.add_noise_to_detections('Datasets/AICity/aicity_annotations.xml', video_length,
    #                                                rescaling_factor=[1.0,1.0], translation_factor=0,
    #                                                prob_discard= 0.0)

    # noisy_gt_boxes = utils.add_noise_to_detections('Datasets/AICity/aicity_annotations.xml', video_length,
    #                                                rescaling_factor=[1.0-noise_range, 1.0+noise_range],
    #                                                translation_factor=0, prob_discard=0.0)

    noisy_gt_boxes = utils.add_noise_to_detections('../Datasets/AICity/aicity_annotations.xml', video_length,
                                                   rescaling_factor=[1.0, 1.0],
                                                   translation_factor=noise_range*10.0, prob_discard=0.0)

    iousPerFrame_list = []
    mAP_list = []
    for random_AP in range(10):
        # detectorName = 'Mask RCNN'
        # detectionFile = 'det_mask_rcnn.txt'
        # detections = utils.getDetections(detectionFileFolder + detectionFile)
        detections = noisy_gt_boxes
        sortedDetections = utils.shuffle_detectionList(detections)
        detectionsPerFrame = utils.getDetectionsPerFrame(sortedDetections)
        print("Compute mAP...")
        mAP = utils.calculate_mAP(groundTruth, detections, IoU_threshold=0.5, have_confidence=False)

        print("Computing IoU over time...")
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
        iousPerFrame_list.append(iousPerFrame)
        mAP_list.append(mAP)
    iousPerFrame_all = np.array(iousPerFrame_list)
    iousPerFrame_mean = np.mean(iousPerFrame_all, axis=0)
    if verbose:
        plt.plot(iousPerFrame)
        plt.ylim((0, 1.0))
        plt.title('Results for iousPerFrame_mean')
        plt.savefig('results/iousPerFrame_mean.png')
        plt.show()

    IoU_mean = np.mean(iousPerFrame_mean)
    mAP_mean = np.mean(mAP_list)

    IoU_mean_list.append(IoU_mean)
    mAP_mean_list.append(mAP_mean)

plt.figure()
plt.plot(noise_range_list, IoU_mean_list)
#plt.xlim((0, 1.0))
plt.ylim((0, 1.0))
plt.title('IoU_mean_list')
plt.savefig('results/IoU_mean_list.png')
plt.show()

plt.figure()
plt.plot(noise_range_list, mAP_mean_list)
#plt.xlim((0, 1.0))
plt.ylim((0, 1.0))
plt.title('Results for mAP_mean_list')
plt.savefig('results/mAP_mean_list.png')
plt.show()

pass
