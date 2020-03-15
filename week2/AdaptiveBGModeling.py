import utils
from sklearn.base import BaseEstimator, ClassifierMixin
from utils_Gaussian import Gaussian_modelling


class AdaptiveBGModeling(BaseEstimator, ClassifierMixin):
    def __init__(self, roi_path, video_path, video_length, video_split_ratio, groundTruth, alpha = 0):
        self.roi_path = roi_path
        self.video_length = video_length
        self.video_split_ratio = video_split_ratio
        self.groundTruth = groundTruth
        self.alpha = alpha
        self.video_path = video_path
        self.detections = []

    def fit(self, X, y=None):
        print("Tested alpha value is " + str(self.alpha))
        assert (self.alpha is not 0), "Alpha is zero"
        _, self.detections = Gaussian_modelling(self.roi_path, self.video_path, alpha=self.alpha, rho=1,
                                                video_length=self.video_length,
                                                video_split_ratio=self.video_split_ratio)
        return self

    def score(self, X, y=None):
        return utils.calculate_mAP(self.groundTruth, self.detections, IoU_threshold=0.5, have_confidence=False)
