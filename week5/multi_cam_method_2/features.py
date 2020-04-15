from skimage.metrics import structural_similarity as ssim
import cv2
import numpy as np
from skimage import feature


def compare_ssim(im1, im2):
    """
    This function compares the structural similarity between two images
    of different shape. They are reshaped to match shape.

    Args:
        im1: image dataset
        im2: image test
    Return:
        distan: the inverse of the SSIM. (The smaller the better.)
    """
    im1 = cv2.cvtColor(cv2.resize(im1, (400, 400)), cv2.COLOR_BGR2GRAY)
    im2 = cv2.cvtColor(cv2.resize(im2, (400, 400)), cv2.COLOR_BGR2GRAY)
    distan = 1 / ssim(im1, im2)
    return distan


def loc_bin_pat(im, bins=50):
    """
    Calculates the histogram of the local binary image of an input image
    Optional parameters to modify:
        - Percentage of cropped section: 0.4
        - Number of points: 4
        - Radius: 1
        - Method: uniform
    Args:
        im: 3 color image
        bins: number of bins of the histogram (10 optimal)

    Rrturn:
        lbp_hist: the local binary pattern histogram of the input image
    """
    im = cv2.resize(im, (400, 400))
    im = cv2.cvtColor(im, cv2.COLOR_BGR2LAB)[..., 0]
    # im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    lbp_im = feature.local_binary_pattern(im, 8, 10, "nri_uniform")
    lbp_im1 = feature.local_binary_pattern(im, 8, 20, "nri_uniform")
    lbp_im2 = feature.local_binary_pattern(im, 8, 30, "nri_uniform")

    hist_im = np.concatenate(
        (
            np.histogram(lbp_im, bins, density=True)[0],
            np.histogram(lbp_im1, bins, density=True)[0],
            np.histogram(lbp_im2, bins, density=True)[0],
        )
    )

    return hist_im

def loc_bin_pat_mr(im, bins=50, splits=(3, 3)):
    """
    Calculates the histogram of the local binary image of an input image
    Optional parameters to modify:
        - Percentage of cropped section: 0.4
        - Number of points: 4
        - Radius: 1
        - Method: uniform
    Args:
        im: 3 color image
        bins: number of bins of the histogram (10 optimal)

    Rrturn:
        lbp_hist: the local binary pattern histogram of the input image
    """

    x_splits, y_splits = splits

    im = cv2.resize(im, (400, 400))
    im = cv2.cvtColor(im, cv2.COLOR_BGR2LAB)[..., 0]

    x_len = int(im.shape[0] / x_splits)
    y_len = int(im.shape[1] / y_splits)

    hist_ims = []
    for i in range(x_splits):
        for j in range(y_splits):
            small_im = im[i * x_len : (i + 1) * x_len, j * y_len : (j + 1) * y_len]

            lbp_im = feature.local_binary_pattern(small_im, 8, 10, "nri_uniform")
            lbp_im1 = feature.local_binary_pattern(small_im, 8, 20, "nri_uniform")
            lbp_im2 = feature.local_binary_pattern(small_im, 8, 30, "nri_uniform")

            hist_im = np.concatenate(
                (
                    np.histogram(lbp_im, bins, density=True)[0],
                    np.histogram(lbp_im1, bins, density=True)[0],
                    np.histogram(lbp_im2, bins, density=True)[0],
                )
            )
            hist_ims.append(hist_im)

    return np.array(hist_ims).ravel()

def compute_hog(image):
    image = cv2.resize(image, (512, 512))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    return feature.hog(image, orientations=8,pixels_per_cell=(64, 64),cells_per_block=(5,5), multichannel=False)



def compute_image_dct(image, block_size=64, num_coefs=10, mask=None):

    image = cv2.resize(image, (512, 512))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    dct_out = []
    for x_block in np.r_[: image.shape[0] : block_size]:
        for y_block in np.r_[: image.shape[1] : block_size]:
            dct_im = cv2.dct(
                image[x_block : x_block + block_size, y_block : y_block + block_size]
                / 255
            )
            dct_coffs = np.concatenate(
                [
                    np.diagonal(dct_im[::-1, :], i)[:: (2 * (i % 2) - 1)]
                    for i in range(1 - dct_im.shape[0], dct_im.shape[0])
                ]
            )[:num_coefs]

            dct_out.append(dct_coffs)

    return np.array(dct_out).ravel()


def compute_mr_histogram(
    img, splits=(1, 1), bins=256, mask=None, sqrt=False, concat=False
):

    x_splits, y_splits = splits
    x_len = int(img.shape[0] / x_splits)
    y_len = int(img.shape[1] / y_splits)

    histograms = []

    for i in range(x_splits):
        for j in range(y_splits):
            small_img = img[i * x_len : (i + 1) * x_len, j * y_len : (j + 1) * y_len]
            small_mask = None
            if mask is not None:
                small_mask = mask[
                    i * x_len : (i + 1) * x_len, j * y_len : (j + 1) * y_len
                ].astype("bool")
            if concat:
                if len(small_img.shape) == 3:
                    small_hist = np.array(
                        [
                            np.histogram(
                                small_img[..., channel][small_mask],
                                bins=bins,
                                density=True,
                            )[0]
                            for channel in range(small_img.shape[2])
                        ]
                    )
                    histograms.append(small_hist.ravel())
                else:
                    raise Exception("Image should have more channels")
            else:
                histograms.append(
                    (
                        np.histogram(small_img[small_mask], bins=bins, density=True)[0]
                    ).ravel()
                )

    histograms = [np.sqrt(hist) if sqrt else hist for hist in histograms]
    return np.concatenate(histograms, axis=0)


def surf_descriptor(image):
    hessianThreshold = 1000
    nOctaves = 12
    nOctaveLayers = 6
    extended = True
    upright = False
    surf = cv2.xfeatures2d.SURF_create(
        hessianThreshold, nOctaves, nOctaveLayers, extended, upright
    )

    image = cv2.resize(image, min((512, 512), image.shape[:2]))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kkpp, descriptors = surf.detectAndCompute(image, None)
    keypoints = np.array([k.pt for k in kkpp])

    return keypoints, descriptors

def sift_descriptor(image, octave_layers=3, nfeatures=0, contrast=0.04, edge=10, sigma=1.6):

    sift = cv2.xfeatures2d.SIFT_create(nfeatures, octave_layers, contrast, edge, sigma)

    image = cv2.resize(image, min((512, 512), image.shape[:2]))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kkpp, descriptors = sift.detectAndCompute(image, None)

    keypoints = np.array([k.pt for k in kkpp])

    return keypoints, descriptors


def orb_descriptor(image):

    orb = cv2.ORB_create()

    image = cv2.resize(image, min((512, 512), image.shape[:2]))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kkpp, descriptors = orb.detectAndCompute(image, None)

    keypoints = np.array([k.pt for k in kkpp])

    return keypoints, descriptors


def filter_matches(matches, similarity_factor=0.7):
    filtered = []
    for orig, match in matches:
        if orig.distance < match.distance * similarity_factor:
            filtered.append(orig)

    return filtered


def calculate_match_dist(matches, min_matches=13):
    if len(matches) < min_matches:
        return np.inf
    else:
        distances = [match.distance for match in matches]
        mean = np.mean(distances)
        std = np.std(distances)
        return len(matches) / min((mean - std), 1)


def compare_keypoints(train_desc, query_desc):
    matches = cv2.BFMatcher(cv2.NORM_L1).knnMatch(train_desc, query_desc, k=2)
    return matches
