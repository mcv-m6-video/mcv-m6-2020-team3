import cv2
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt

import numpy as np

from skimage import measure



def Gaussian_modelling(roi_path, video_path, alpha, rho, video_length = 2141, video_split_ratio = 0.25):
    roi = cv2.cvtColor(cv2.imread(roi_path), cv2.COLOR_BGR2GRAY)

    video_first_part, video_second_part, divide_frame = \
        read_video_and_divide(video_path, video_length=video_length, video_split_ratio=video_split_ratio)

    video_first_part_mean, video_first_part_std = calculate_mean_std_first_part_video(video_first_part)

    print('Extracting foreground...')
    foreground_second_part = calculate_mask(roi, video_second_part, video_first_part_mean, video_first_part_std, alpha)

    detections = find_detections(foreground_second_part, first_frame_id=divide_frame + 1)

    print('Finish Extracting foreground...foreground_second_part.shape = {}', foreground_second_part.shape)

    return foreground_second_part, detections

def read_video_and_divide(video_path, video_length, video_split_ratio):
    """
    read video, transform into gray and divide
    """
    print("begin reading video")
    video = cv2.VideoCapture(video_path)

    divide_frame = int(video_length*video_split_ratio)

    for i in tqdm(range(video_length)):
        if video.isOpened():
            valid, frame = video.read()
        else:
            raise Exception("video.isOpened() = false")
        if not valid:
            raise Exception("frame_valid = false")

        if i == 0:
            video_first_part = np.zeros((divide_frame, frame.shape[0], frame.shape[1]))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            video_first_part[i, :, :] = frame
        elif i < divide_frame:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            video_first_part[i, :, :] = frame
        elif i == divide_frame:
            video_second_part = np.zeros((video_length - divide_frame, frame.shape[0], frame.shape[1]))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            video_second_part[i - divide_frame, :, :] = frame
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            video_second_part[i-divide_frame, :, :] = frame

    print("finish reading video")

    return video_first_part, video_second_part, divide_frame


def calculate_mean_std_first_part_video(video_first_part):
    """
    read video, and calculate the mean of first 25%
    """
    print ("read video, and calculate the mean of first pard")
    video_first_part_mean = video_first_part.mean(axis=0)
    video_first_part_std = video_first_part.std(axis=0)

    print("finish calculating the mean and std")

    return video_first_part_mean, video_first_part_std


def calculate_mask(roi, video_second_part, frame_mean, frame_std, alpha):
    """
    calculate_mask
    """
    foreground_second_part = np.zeros(video_second_part.shape)
    for i in range(video_second_part.shape[0]):
        frame = video_second_part[i, :, :]
        foreground_1 = foreground_gaussian(frame, frame_mean, frame_std, alpha)
        foreground_roi = roi * foreground_1
        foreground_filtered = morphological_filter(foreground_roi)
        foreground_second_part[i, :, :] = foreground_filtered

    return foreground_second_part


def find_detections(foreground_second_part, first_frame_id, min_h=80, max_h=500, min_w=100,
                    max_w=600, min_ratio=0.1, max_ratio=10.0):
    detections = []
    for i in range(foreground_second_part.shape[0]):
        frame_id = i + first_frame_id
        mask = foreground_second_part[i, :, :]
        label_image = measure.label(mask)
        regions = measure.regionprops(label_image)
        detection = {}
        for region in regions:
            bbox = region.bbox
            if filter_region(bbox, min_h, max_h, min_w, max_w, min_ratio, max_ratio):
                box_h = bbox[2] - bbox[0]
                box_w = bbox[3] - bbox[1]
                detection['frame'] = frame_id
                detection['left'] = bbox[1]
                detection['top'] = bbox[0]
                detection['width'] = box_w
                detection['height'] = box_h
                detections.append(detection)

    return detections


def filter_region(bbox, min_h, max_h, min_w, max_w, min_ratio, max_ratio):
    box_h = bbox[2] - bbox[0]
    box_w = bbox[3] - bbox[1]
    if box_h > max_h or box_w > max_w:
        return False
    if box_h < min_h or box_w < min_w:
        return False
    if (box_h / box_w) > max_ratio or (box_h / box_w) < min_ratio:
        return False
    else:
        return True


def foreground_gaussian(img, model_mean, model_std, alpha):
    foreground = abs(img - model_mean) >= alpha * (model_std + 2)
    return foreground


def fill_holes(mask):
    im_floodfill = mask.astype(np.uint8).copy()
    h, w = im_floodfill.shape[:2]
    filling_mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(im_floodfill, filling_mask, (0, 0), 1)
    return mask.astype(np.uint8) | (1 - im_floodfill)


def filter_noise(mask):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (50, 50))
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
    mask = cv2.medianBlur(mask, 7)
    return mask

# def fill_holes(mask):
#     im_floodfill = mask.astype(np.uint8).copy()
#     h, w = im_floodfill.shape[:2]
#     filling_mask = np.zeros((h + 2, w + 2), np.uint8)
#     cv2.floodFill(im_floodfill, filling_mask, (0, 0), 1)
#
#     return mask.astype(np.uint8) | (1 - im_floodfill)
#
# def filter_noise(mask):
#     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
#     kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
#     mask = cv2.medianBlur(mask, 9)
#     #mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
#     mask = cv2.erode(mask, kernel, iterations=1)
#     mask = cv2.dilate(mask, kernel2, iterations=1)
#     mask = cv2.erode(mask, kernel)
#     return mask

def morphological_filter(mask):
    """
    Apply morphological operations to prepare pixel candidates to be selected as
    a traffic sign or not.
    """

    plt.imshow(mask)
    plt.show()
    mask_filled = fill_holes(mask)
    plt.imshow(mask_filled)
    plt.show()
    mask_filtered = filter_noise(mask_filled)
    plt.imshow(mask_filtered)
    plt.show()
    return mask_filtered

