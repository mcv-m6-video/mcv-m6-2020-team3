import cv2
import matplotlib.pyplot as plt 
import numpy as np 
import os

from utils import read_annotations, addBboxesToFrames, calculate_mAP

def adaptive_modelling_background(video_path, alpha_factor=1.5, rho_factor=0.1, mask_roi=None, path_to_save=None, color_space='GRAY'):
    
    capture_frames = cv2.VideoCapture(video_path)
    num_frames = capture_frames.get(cv2.CAP_PROP_FRAME_COUNT)
    frame_width = int(capture_frames.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture_frames.get(cv2.CAP_PROP_FRAME_HEIGHT))
    scaling_factor = 0.5
    # Configuration for Training frames
    flag = True
    training_frames = num_frames * 0.25

    if color_space == 'GRAY':
        frames_acc = np.zeros((frame_height, frame_width), dtype='float')
        var_acc = np.zeros((frame_height, frame_width), dtype='float')
    
    elif color_space == 'YUV':
        frames_acc = np.zeros((frame_height, frame_width, 3), dtype='float')
        var_acc = np.zeros((frame_height, frame_width, 3), dtype='float')
    else:
        print("wrong color space")
        return


    iter_frame = 0
    

    while flag and iter_frame<training_frames:

        flag, frames = capture_frames.read()
        iter_frame = int(capture_frames.get(cv2.CAP_PROP_POS_FRAMES))
        
        if color_space == 'GRAY':
            frames = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)
        elif color_space == 'YUV':
            frames = cv2.cvtColor(frames, cv2.COLOR_BGR2YUV)
        frames_acc += frames

        # Mean and variance computation for the training frames
        print('Mean and variance computation for the training frames...')
        mean_image = frames_acc / training_frames

        # set the current frame to 0 and computing the variance

        capture_frames.set(cv2.CAP_PROP_POS_FRAMES, 0)

        iter_frame = int(capture_frames.get(cv2.CAP_PROP_POS_FRAMES))
        flag = True

        while flag and iter_frame<training_frames:
            flag, frames = capture_frames.read()
            
            iter_frame = int(capture_frames.get(cv2.CAP_PROP_POS_FRAMES))
            if color_space == 'GRAY':
                frames = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)
            elif color_space == 'YUV':
                frames = cv2.cvtColor(frames, cv2.COLOR_BGR2YUV)
            var_acc += (frames - mean_image) ** 2

        variance_image = (var_acc/training_frames)

        ## Extract the foreground from the rest 75 % of the frames 

        masks = []
        
        
        detections = dict()

        while flag:
            flag, srcFrame = capture_frames.read()
            if not flag: continue
            iter_frame = int(capture_frames.get(cv2.CAP_PROP_POS_FRAMES))
            print('frame: ', iter_frame)

            frames = srcFrame
            if color_space == 'GRAY':
                frame = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)
            elif color_space == 'YUV':
                frame = cv2.cvtColor(frames, cv2.COLOR_BGR2YUV)

            probability_Image = np.absolute(frames - mean_image) - alpha_factor * (np.sqrt(variance_image)+ 2)
            ret, foreground_Mask = cv2.threshold(probability_Image, 0, 255, cv2.THRESH_BINARY)
            foreground_Mask = np.uint8(foreground_Mask)

            # cv2.imshow("window", foreground_Mask)
            # cv2.waitKey()


            if color_space != 'GRAY':
                maskYU = cv2.bitwise_and(foreground_Mask[:,:,0], foreground_Mask[:,:,1])
                maskYV = cv2.bitwise_and(foreground_Mask[:,:,0], foreground_Mask[:,:,2])
                foreground_Mask = cv2.bitwise_or(maskYU[:,:], maskYV[:,:])
            
            if mask_roi is not None:
                foreground_Mask = cv2.bitwise_and(foreground_Mask[:,:], mask_roi[:,:])
            elif mask_roi is not None:
                foreground_Mask = cv2.bitwise_and(foreground_Mask[:,:], mask_roi[:,:])

            if path_to_save is not None:
                if not os.path.exists(path_to_save):
                    os.makedirs(path_to_save)
                cv2.imwrite(os.path.join(path_to_save, str(iter_frame) + '.png'), foreground_Mask.astype('uint8'))


            # cv2.imshow("window", foreground_Mask)
            # cv2.waitKey()

            mean_image = (1-rho_factor)*mean_image + rho_factor*frames
            variance_image = np.sqrt((1-rho_factor)*np.power(variance_image, 2) + rho_factor*np.power((frames-mean_image), 2))

            # cv2.imshow("window", foreground_Mask)
            # cv2.waitKey()
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)) #get the elliptical kernel 
            mask_Resized = cv2.resize(foreground_Mask, (0, 0), fx=0.4, fy=0.4) #scaling the image
            # maskResized = foreground_Mask
            # cv2.imshow("window", maskResized)
            # cv2.waitKey()
            # apply morphological opening to the result to remove the noises.
            opened_Mask = cv2.morphologyEx(mask_Resized, cv2.MORPH_CLOSE, kernel)
            opened_Mask = cv2.resize(opened_Mask, (frame_width, frame_height))
            # cv2.imshow("window", openedMask)
            # cv2.waitKey()
            # openedMask = foreground_Mask
            opened_Mask = np.uint8(opened_Mask)

            nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(opened_Mask, 4, cv2.CV_32S)

            min_size = 1000
            max_size = 1000000
            
            for i in range(nlabels):
                
                if stats[i][4] >= min_size and stats[i][4] <= max_size and (
                    stats[i, cv2.CC_STAT_WIDTH] / stats[i, cv2.CC_STAT_HEIGHT]) < 2.8:

                    bbox = [stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]]
                    cv2.rectangle(srcFrame, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (0, 0, 255), 7)

                    # add the detections in the dictionary
                    content = bbox
                    content.extend([1.])  # confidence
                    if str(iter_frame) in detections.keys():
                        detections[str(iter_frame)].append(content)
                    else:
                        detections[str(iter_frame)] = [content]

            plt.imshow(cv2.resize(cv2.cvtColor(srcFrame, cv2.COLOR_BGR2RGB), (0, 0), fx=scaling_factor, fy=scaling_factor))
            plt.show(block=False)
            plt.pause(0.05)
            plt.clf()

            # plt.imshow(cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (0,0), fx=rescaling_factor, fy=rescaling_factor))

    capture_frames.release()
    return detections, training_frames



def main(alpha, rho):
    # path = '../../Datasets/AICity_data/AICity_data/train/S03/c010/vdo.avi'
    # path_to_save_adap = '../../Datasets/AICity_data/AICity_data/train/S03/c010/maskAdap/'
    # path_roi = '../../Datasets/AICity_data/AICity_data/train/S03/c010/roi.jpg'

    path = './Datasets/AICity_data/AICity_data/train/S03/c010/vdo.avi'
    path_to_save_adap = './Datasets/AICity_data/AICity_data/train/S03/c010/maskAdap/'
    path_roi = './Datasets/AICity_data/AICity_data/train/S03/c010/roi.jpg'

    mask_roi = cv2.imread(path_roi, cv2.IMREAD_GRAYSCALE)

    bboxes_detections, training_frames = adaptive_modelling_background(path, alpha_factor=alpha, rho_factor=rho, mask_roi=mask_roi, path_to_save=None
                                                             ,color_space='YUV')

    # bboxes_gt, num_instances_gt = get_gt_bboxes()
    groundTruth = read_annotations('/Datasets/aicity_annotations.xml', 2141)
    # get the number of instances in the validation split (needed to calculate the number of FN and the recall)
    # num_instances_validation = 0
    # for key in bboxes_gt.keys():
    #     if int(key) > trainingFrames:
    #         num_instances_validation += len(bboxes_gt[key])
    IoU_threshold = 0.5
    
    mAP=calculate_mAP(groundTruth, bboxes_detections, IoU_threshold=0.5, have_confidence = True, verbose = False)

    
    # TP, FP, FN, scores = evaluation_detections(threshold, bboxes_gt, bboxes_detected, num_instances_validation, trainingFrames)
    # print("tp: ", TP, "fp: ", FP, "fn: ", FN)
    # pr, pinterps, idxs_interpolations, mAP, APs = compute_map(scores, num_instances_validation)

    print("map: ", mAP)
    # plot_pr(pr, threshold, pinterps, idxs_interpolations, APs)  # plot mAP

    # bboxes_gt, num_instances_gt = get_gt_bboxes()  # load the bboxes from the gt again
    # show_bboxes(path, bboxes_gt, bboxes_detected)  # show the bounding boxes


if __name__ == "__main__":
    search_space = [(3, 0.5), (4, 0.5), (5, 0.5), (6, 0.5), (7, 0.5)]
    # I've commented all the prints for now so I can see the scores outputed for every configuration without too
    # much stuff in between
    for config in search_space:
        print("trying alpha - rho: ", config[0], " - ", config[1])
        main(config[0], config[1])



































































