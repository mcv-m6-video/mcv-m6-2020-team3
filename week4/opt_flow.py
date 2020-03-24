import cv2
import numpy as np
import time
import pyflow

def draw_flow(img, flow, step=8):
    h, w = img.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1).astype(int)
    fx, fy = flow[y,x].T
    lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.polylines(vis, lines, 0, (0, 255, 0))
    for (x1, y1), (x2, y2) in lines:
        cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
    return vis

def read_file(path):
    """Read an optical flow map from disk
    Optical flow maps are stored in disk as 3-channel uint16 PNG images,
    following the method described in the KITTI optical flow dataset 2012
    (http://www.cvlibs.net/datasets/kitti/eval_stereo_flow.php?benchmark=flow).
    Returns:
      numpy array with shape [height, width, 3]. The first and second channels
      denote the corresponding optical flow 2D vector (u, v). The third channel
      is a mask denoting if an optical flow 2D vector exists for that pixel.
      Vector components u and v values range [-512..512].
    """
    data = cv2.imread(path, -1).astype('float32')
    result = np.empty(data.shape, dtype='float32')
    result[:,:,0] = (data[:,:,2] - 2**15) / 64
    result[:,:,1] = (data[:,:,1] - 2**15) / 64
    result[:,:,2] = data[:,:,0]

    return result
def coarse2Fine(prev_frame, curr_frame, viz):
    prev_frame_gray_nc = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    curr_frame_gray_nc = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

    prev_frame_gray = prev_frame_gray_nc[:, :, np.newaxis]
    curr_frame_gray = curr_frame_gray_nc[:, :, np.newaxis]

    prev_array = np.asarray(prev_frame_gray)
    curr_array = np.asarray(curr_frame_gray)

    prev_array = prev_array.astype(float) / 255.
    curr_array = curr_array.astype(float) / 255.

    # Flow Options:
    alpha = 0.012
    ratio = 0.75
    minWidth = 20
    nOuterFPIterations = 7
    nInnerFPIterations = 1
    nSORIterations = 30
    colType = 1  # 0 or default:RGB, 1:GRAY (but pass gray image with shape (h,w,1))

    s = time.time()
    u, v, im2W = pyflow.coarse2fine_flow(
        prev_array, curr_array, alpha, ratio, minWidth, nOuterFPIterations, nInnerFPIterations,
        nSORIterations, colType)
    e = time.time()
    print('Time Pyflow Taken: %.2f seconds for image of size (%d, %d, %d)' % (
        e - s, prev_array.shape[0], prev_array.shape[1], prev_array.shape[2]))
    flow = np.concatenate((u[..., None], v[..., None]), axis=2)

    array_ones = np.ones((u.shape[0], u.shape[1]))
    flow_return = np.ndarray((u.shape[0], u.shape[1], 3))
    flow_return[:,:,0] = u
    flow_return[:,:,1] = v
    flow_return[:,:,2] = array_ones
    #plot
    if viz:
        hsv = np.zeros(prev_frame.shape, dtype=np.uint8)
        hsv[:, :, 0] = 255
        hsv[:, :, 1] = 255
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        cv2.imwrite('output/pyflow.png', rgb)
        cv2.imshow("optical flow", rgb)
        cv2.waitKey()
        arrows = draw_flow(curr_frame_gray_nc, flow)
        cv2.imwrite('output/pyflow_arrows.png', arrows)
        cv2.imshow('pyflow', arrows)
        cv2.waitKey()

    return flow, flow_return

def farneback(frame1, frame2 ,viz):

    prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    s = time.time()
    flow = cv2.calcOpticalFlowFarneback(prvs, next, flow=None,
                                        pyr_scale=0.5, levels=3,
                                        winsize=15, iterations=3,
                                        poly_n=5, poly_sigma=1.2, flags=0)
    e = time.time()
    print('Time farneback Taken: %.2f seconds for image of size (%d, %d, %d)' % (
        e - s, frame1.shape[0], frame1.shape[1], frame1.shape[2]))
    # Plot
    if viz:
        hsv = np.zeros_like(frame1)
        hsv[..., 1] = 255
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        cv2.imwrite('output/optical_farneback.png', bgr)
        cv2.imshow("optical farneback flow", bgr)
        cv2.waitKey()
        arrows = draw_flow(next, flow)
        cv2.imwrite('output/optical_farneback_arrows.png', arrows)
        cv2.imshow('optical farneback', arrows)
        cv2.waitKey()
    return flow

def lucas_kanade_denso(old_frame, frame, viz):
    # params for corner detection
    feature_params = dict(maxCorners=100,
                          qualityLevel=0.3,
                          minDistance=7,
                          blockSize=7)

    # Parameters for lucas kanade optical flow
    lk_params = dict(winSize=(15, 15),
                     maxLevel=5,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                               10, 0.03))
    old_gray = cv2.cvtColor(old_frame,
                            cv2.COLOR_BGR2GRAY)
    height = old_frame.shape[0]
    width = old_frame.shape[1]
    frame_gray = cv2.cvtColor(frame,
                              cv2.COLOR_BGR2GRAY)
    s = time.time()
    # calculate all points used for find dense flow
    all_pixels = np.nonzero(frame_gray)[::-1]
    all_pixels = tuple(zip(*all_pixels))
    all_pixels = np.vstack(all_pixels).reshape(-1, 1, 2).astype("float32")
    # calculate optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray,
                                           frame_gray,
                                           all_pixels, None,
                                           **lk_params)
    e = time.time()
    print('Time Lucas Kanade Dense Taken: %.2f seconds for image of size (%d, %d, %d)' % (
        e - s, frame_gray.shape[0], frame_gray.shape[1], 1))

    all_pixels = all_pixels.reshape(height, width, 2)
    p1 = p1.reshape(height, width, 2)
    # Flow vector calculated by subtracting new pixels by old pixels
    flow = p1 - all_pixels
    if viz:
        hsv = np.zeros_like(frame)
        hsv[..., 1] = 255
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        cv2.imwrite('output/optical_l_k_dense.png', bgr)
        cv2.imshow("optical L-K dense flow", bgr)
        cv2.waitKey()
        arrows = draw_flow(frame_gray, flow, step=8)
        cv2.imwrite('output/optical_l_k_dense_arrows.png', arrows)
        cv2.imshow('optical L-K dense', arrows)
        cv2.waitKey()
    return flow