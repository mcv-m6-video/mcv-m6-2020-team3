import numpy as np


def flow_metrics(pred, gt):
    flowExist  = (gt[: ,: ,2] == 1)
    pred_flow  = pred[flowExist]
    gt_flow    = gt[flowExist]
    # print(flowExist.shape)
    img_err = np.zeros(shape=gt[: ,: ,1].shape)

    err = gt_flow[: ,:2] - pred_flow[: ,:2]

    squared_err = np.sum(err**2, axis=1)
    vect_err = np.sqrt(squared_err)
    hit = vect_err < 3.0
    img_err[flowExist] = vect_err

    msen = np.mean(vect_err)
    pepn = 100 * (1 - np.mean(hit))

    return msen, pepn, img_err, vect_err

