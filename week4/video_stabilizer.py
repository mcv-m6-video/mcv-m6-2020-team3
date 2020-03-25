import numpy as np
import cv2
import matplotlib.pyplot as plt
import imageio
import pickle
from scipy.signal import savgol_filter

from block_matching import block_matching_optical_flow


def plot_traj(x, y, name):
    # Data for plotting
    fig, ax = plt.subplots()
    ax.plot(x, y)
    ax.set(xlabel='Frame number', ylabel='pixels', title=name)
    ax.grid()
    fig.savefig(name + ".png")
    plt.show()


def plot_before_and_after(x, y1, y2, name):
    # Data for plotting
    fig, ax = plt.subplots()
    ax.plot(x, y1, label='Origin')
    ax.plot(x, y2, label='After smoothing')
    ax.set(xlabel='Frame number', ylabel='pixels', title=name)
    ax.grid()
    fig.savefig(name + ".png")
    plt.show()


def video_stabilization(sequence):

    prev = sequence[0]

    seq_stabilized_1 = []
    seq_stabilized_1.append(prev)

    traj_x = 0.0
    traj_y = 0.0

    traj_x_record = []
    traj_y_record = []
    traj_x_record.append(traj_x)
    traj_y_record.append(traj_y)

    flag = False
    if flag:
        # 1 - Get previous to current frame transformation (dx, dy, da) for all frames
        for i in range(1, len(sequence)):

            next = sequence[i]

            vector_field = block_matching_optical_flow(prev, next)
            vector_u = vector_field[:, :, 1]
            vector_v = vector_field[:, :, 0]

            vector_u_order, times = np.unique(vector_u, return_counts=True)
            vector_u_median = vector_u_order[times.argmax()]
            vector_v_order, times = np.unique(vector_v, return_counts=True)
            vector_v_median = vector_v_order[times.argmax()]

            dx = vector_u_median
            dy = vector_v_median


        # 2 - Accumulate the transformations to get the image trajectory
            traj_x = traj_x + dx
            traj_y = traj_y + dy

            traj_x_record.append(traj_x)
            traj_y_record.append(traj_y)

            prev = next
            print("Trajectory {}: x={}, y={}". format(i, traj_x,traj_y))

            H = np.array([[1, 0, -traj_x], [0, 1, -traj_y]], dtype=np.float32)
            next_stabilized = cv2.warpAffine(next, H, (next.shape[1], next.shape[0]))  # translation + rotation only

            seq_stabilized_1.append(next_stabilized)

        imageio.mimsave('seq_stabilized_1.gif', seq_stabilized_1)

        with open("traj_ori.pkl", 'wb') as f:
            pickle.dump([traj_x_record, traj_y_record], f)
            f.close()
    else:
        with open("traj_ori.pkl", 'rb') as p:
            traj_x_record, traj_y_record = pickle.load(p)
            p.close()

    # for i in range(len(traj_x_record)):
    #     traj_x_record[i] = -traj_x_record[i]

    plot_traj(np.arange(len(traj_x_record)), traj_x_record, 'Trajectory x')
    plot_traj(np.arange(len(traj_y_record)), traj_y_record, 'Trajectory y')

    smooth_trajs = []
    #  3 - Smooth out the trajectory using an averaging window
    smoothed = savgol_filter(np.array(traj_x_record), 45, 3)
    traj_x_record_smooth = np.around(smoothed).tolist()
    smoothed = savgol_filter(np.array(traj_y_record), 45, 3)
    traj_y_record_smooth = np.around(smoothed).tolist()

    # Data for plotting
    plot_before_and_after(np.arange(len(traj_x_record)), traj_x_record, traj_x_record_smooth, 'traj_x_smooth')
    plot_before_and_after(np.arange(len(traj_y_record)), traj_y_record, traj_y_record_smooth, 'traj_y_smooth')

    seq_stabilized_2 = []
    # 4 - Apply the new transformation to the video
    for i in range(len(sequence)):
        frame = sequence[i]
        # H = np.array([[np.cos(news[i].a), - np.sin(news[i].a), -news[i].x], [np.sin(news[i].a), np.cos(news[i].a), -news[i].y]],
        #            dtype=np.float32)
        delta_x = traj_x_record_smooth[i] - traj_x_record[i]
        delta_y = traj_y_record_smooth[i] - traj_y_record[i]
        H = np.array([[1, 0, delta_x], [0, 1, delta_y]],
                     dtype=np.float32)
        frame_stabilized = cv2.warpAffine(frame, H, (frame.shape[1], frame.shape[0]))
        seq_stabilized_2.append(frame_stabilized)

    imageio.mimsave('seq_stabilized_2.gif', seq_stabilized_2)

    return seq_stabilized_2


def read_mp4_sequences(path):
    videoCapture = cv2.VideoCapture(path)

    fps = videoCapture.get(cv2.CAP_PROP_FPS)
    size = (int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fNUMS = videoCapture.get(cv2.CAP_PROP_FRAME_COUNT)

    img_seq = []
    success, frame = videoCapture.read()
    while success:
        ratio = 0.125
        resized = cv2.resize(frame, (int(ratio * size[0]), int(ratio * size[1])), interpolation=cv2.INTER_AREA)
        # show video
        # cv2.imshow('windows', resized)
        # cv2.waitKey(int(1000.0 / int(fps)))
        img_seq.append(resized)
        success, frame = videoCapture.read()

    videoCapture.release()

    img_seq_decrease = [img_seq[i] for i in range(0, len(img_seq), 3)]
    return img_seq_decrease


if __name__ == "__main__":

    print("Video Stabilization with block matching")
    sequences = read_mp4_sequences('VID20200325141005.mp4')
    sequences = sequences[10:]
    imageio.mimsave('before.gif', sequences)
    seq_stabilized = video_stabilization(sequences)



