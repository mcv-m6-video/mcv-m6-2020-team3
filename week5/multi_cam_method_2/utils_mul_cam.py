import copy
import numpy as np
import cv2
from tqdm import tqdm
import imageio

def merge_tracks(old_track_list):
    old_track_list.sort(key=lambda x: x.id)
    copy_of_tracks = copy.deepcopy(old_track_list)
    new_track_list = []
    last_id = -1
    new_track = []
    for track_one in copy_of_tracks:
        if track_one.id != last_id:
            if last_id != -1:
                new_track_list.append(new_track)
            new_track = track_one
        else:
            new_track.detections.extend(track_one.detections)
        last_id = track_one.id
    return new_track_list


def center_of_detection(detection):
    return (int(detection['left'] + 0.5 * detection['width']), int(detection['top'] + 0.5 * detection['height']))


def write_one_frame_multi_cam(detections_tracks, frame_id, frameMat, color, offset_line = 0):
    """
    tool for addTracksToFrames
    :param detections_tracks: this can be detections or ground truth
    :param frame_id: which frame
    :param frameMat: frame picture
    :param color: rectangle and line color
    :return: frameMat
    """
    thick_line = 10
    for track_one in detections_tracks:
        index = 0
        flag_shoot = False

        count_det = 0
        centerPoint_list = []
        for index, detection in enumerate(track_one.detections):
            # write the rectangle
            if detection['frame'] == frame_id:
                startPoint = (int(detection['left']), int(detection['top']))
                endPoint = (int(startPoint[0] + detection['width']), int(startPoint[1] + detection['height']))
                frameMat = cv2.rectangle(frameMat, startPoint, endPoint, color, 2)
                flag_shoot = True
                count_det = count_det + 1
                centerPoint = center_of_detection(detection)
                centerPoint_list.append(centerPoint)
        if count_det > 1:
            for i in range(count_det):
                for j in range(i+1, count_det):
                    startPoint = centerPoint_list[i]
                    startPoint = (startPoint[0], startPoint[1] + offset_line)
                    endPoint = centerPoint_list[j]
                    endPoint = (endPoint[0], endPoint[1] + offset_line)
                    frameMat = cv2.line(frameMat, startPoint, endPoint, color, thick_line)
        # if (frame_id >= track_one.detections[0]['frame']) and (frame_id <= track_one.detections[-1]['frame']):
        #     shoot_index = index
        #     # write the line
        #     for index, detection in enumerate(track_one.detections):
        #         if detection['frame'] > frame_id:
        #             break
        #         endPoint = center_of_detection(detection)
        #         if index != 0:
        #             frameMat = cv2.line(frameMat, startPoint, endPoint, color, thick_line)
        #         startPoint = endPoint
        #     # if flag_shoot == False, draw the line with middle point instead end point
        #     if flag_shoot == False:
        #         endPoint = center_of_detection(track_one.detections[index])
        #         endframe = track_one.detections[index]['frame']
        #         startPoint = center_of_detection(track_one.detections[index-1])
        #         startframe = track_one.detections[index-1]['frame']
        #         ratio = (frame_id-startframe)/(endframe-startframe)
        #         g = lambda x1, x2, r: int(x1 + r * (x2-x1))
        #         middlepoint = (g(startPoint[0], endPoint[0], ratio), g(startPoint[1], endPoint[1], ratio))
        #         frameMat = cv2.line(frameMat, startPoint, middlepoint, color, thick_line)

    return frameMat


def offset_tracks(tracks, offset):
    for track_one in tracks:
        for det in track_one.detections:
            offset_cam = offset[det['cam']]
            det['left'] = det['left'] + offset_cam[0]
            det['top'] = det['top'] + offset_cam[1]
    return tracks


def write_offset_images(framesPath, offset, camera_list, timestamp, fps_ratio, video_length_list, frame_id):
    # create a big frame
    img = np.zeros((4000, 9000, 3), np.uint8)
    # use white to fill.
    img.fill(255)

    for cam in camera_list:
        real_frame = int((frame_id-timestamp[cam])/fps_ratio[cam])
        offset_cam = offset[cam]
        if real_frame < 1 or real_frame > video_length_list[cam]:
            continue
        image = cv2.imread("{}{}/frames/{}.jpg".format(framesPath, cam, str(real_frame).zfill(5)))
        img[offset_cam[1]:(offset_cam[1]+image.shape[0]), offset_cam[0]:(offset_cam[0]+image.shape[1]), :] = \
            image[:, :, :]
    return img

def addTracksToFrames_multi_cam(framesPath, detections_tracks_ori, tracks_gt_list_ori, offset, camera_list,
                                timestamp, fps_ratio, video_length_list, start_frame = 1, end_frame = 2422,
                                name = "test"):
    """
    write the video of the tracking result in the format .avi
    :param framesPath: path of frames
    :param detections_tracks: detections in list of Track. Detections in Track should be sorted.
    :param tracks_gt_list: ground truth in list of Track. Detections in Track should be sorted.
    :param name: name of video.
    :return: None
    """
    detections_tracks = copy.deepcopy(detections_tracks_ori)
    tracks_gt_list = copy.deepcopy(tracks_gt_list_ori)
    detections_tracks = offset_tracks(detections_tracks, offset)
    tracks_gt_list = offset_tracks(tracks_gt_list, offset)

    # size = (9000, 4000)
    size = (2250, 1000)
    fps = 10
    out = cv2.VideoWriter(name + '.avi', cv2.VideoWriter_fourcc(*'DIVX'), fps, size)

    for frame_id in tqdm(range(start_frame, end_frame)):
        frameMat = write_offset_images(framesPath, offset, camera_list, timestamp, fps_ratio,
                                       video_length_list, frame_id)

        color_detection = (0, 0, 255)
        write_one_frame_multi_cam(detections_tracks, frame_id, frameMat, color_detection)
        color_gt = (255, 0, 0)
        offset_line = 5
        write_one_frame_multi_cam(tracks_gt_list, frame_id, frameMat, color_gt, offset_line)
        resized = cv2.resize(frameMat, size, interpolation=cv2.INTER_AREA)
        out.write(resized)
    out.release()


def addTracksToFrames_multi_cam_gif(framesPath, detections_tracks_ori, tracks_gt_list_ori, offset, camera_list,
                                timestamp, fps_ratio, video_length_list, start_frame = 1, end_frame = 2422,
                                name = "test"):
    """
    write the video of the tracking result in the format .avi
    :param framesPath: path of frames
    :param detections_tracks: detections in list of Track. Detections in Track should be sorted.
    :param tracks_gt_list: ground truth in list of Track. Detections in Track should be sorted.
    :param name: name of video.
    :return: None
    """
    detections_tracks = copy.deepcopy(detections_tracks_ori)
    tracks_gt_list = copy.deepcopy(tracks_gt_list_ori)
    detections_tracks = offset_tracks(detections_tracks, offset)
    tracks_gt_list = offset_tracks(tracks_gt_list, offset)

    # size = (9000, 4000)
    size = (1125, 500)
    fps = 10
    images = []
    skip = 3

    for frame_id in tqdm(range(start_frame, end_frame)):
        frameMat = write_offset_images(framesPath, offset, camera_list, timestamp, fps_ratio,
                                       video_length_list, frame_id)

        color_detection = (0, 0, 255)
        write_one_frame_multi_cam(detections_tracks, frame_id, frameMat, color_detection)
        color_gt = (255, 0, 0)
        offset_line = 5
        write_one_frame_multi_cam(tracks_gt_list, frame_id, frameMat, color_gt, offset_line)
        resized = cv2.resize(frameMat, size, interpolation=cv2.INTER_AREA)
        if frame_id%skip == 0:
            images.append(resized)
    imageio.mimsave(name + '.gif', images)



def sort_track(tracks):
    for track_one in tracks:
        track_one.detections.sort(key=lambda x: x['frame'])