import numpy as np
import math
from joblib import Parallel, delayed

"""
Metrics functions for points extraction
Adapted from the AI4CMA contest metrics code
https://CriticalMinerals.darpa.mil/Files/AI4CMA_Challenge_2_Metric.ipynb
"""

# MIN_VALID_RANGE = buffering distance around points -- as a percent of an image's diagonal pixel dimension
MIN_VALID_RANGE = 0.25
DIFFICULT_WEIGHT = 0.7
SET_FALSE_AS = "hard"
BINARY_PIXEL_VAL = 1


def overlap_distance_calculate(
    mat_true, mat_pred, min_valid_range=0.1, parallel_workers=1
):
    """
    mat_true, mat_pred: 2d matrices, with 0s and 1s only
    min_valid_range: the maximum distance in % of the largest size of the image (diagonal)
        between a predicted pixel vs. a true one that will be considered
        as valid to include in the scoring.
    calculate_distance: when True this will not only calculate overlapping pixels
        but also the distances between nearesttrue and predicted pixels
    """

    lowest_dist_pairs = []
    points_done_pred = set()
    points_done_true = set()

    # first calculate the overlapping pixels
    mat_overlap = mat_pred * mat_true
    # for x_true, y_true in tqdm(np.argwhere(mat_overlap==1)):
    for x_true, y_true in np.argwhere(mat_overlap == 1):
        lowest_dist_pairs.append((((x_true, y_true), (x_true, y_true)), 0.0))
        points_done_true.add((x_true, y_true))
        points_done_pred.add((y_true, x_true))
    # print('len(lowest_dist_pairs) by overlapping only:', len(lowest_dist_pairs))

    diagonal_length = math.sqrt(
        math.pow(mat_true.shape[0], 2) + math.pow(mat_true.shape[1], 2)
    )
    min_valid_range = int((min_valid_range * diagonal_length) / 100)  # in pixels
    # print('calculated pixel min_valid_range:', min_valid_range)

    def nearest_pixels(x_true, y_true):
        result = []
        # find all the points in pred withing min_valid_range rectangle
        mat_pred_inrange = mat_pred[
            max(x_true - min_valid_range, 0) : min(
                x_true + min_valid_range, mat_true.shape[1]
            ),
            max(y_true - min_valid_range, 0) : min(
                y_true + min_valid_range, mat_true.shape[0]
            ),
        ]
        for x_pred_shift, y_pred_shift in np.argwhere(mat_pred_inrange == 1):
            y_pred = max(y_true - min_valid_range, 0) + y_pred_shift
            x_pred = max(x_true - min_valid_range, 0) + x_pred_shift
            if (x_pred, y_pred) in points_done_pred:
                continue
            # calculate eucledean distances
            dist_square = math.pow(x_true - x_pred, 2) + math.pow(y_true - y_pred, 2)
            result.append((((x_true, y_true), (x_pred, y_pred)), dist_square))
        return result

    candidates = [
        (x_true, y_true)
        for x_true, y_true in np.argwhere(mat_true == 1)
        if (x_true, y_true) not in points_done_true
    ]
    distances = Parallel(n_jobs=parallel_workers)(
        delayed(nearest_pixels)(x_true, y_true) for x_true, y_true in candidates
    )
    distances = [item for sublist in distances for item in sublist]  # type: ignore

    # sort based on distances
    distances = sorted(distances, key=lambda x: x[1])

    # find the lowest distance pairs
    for ((x_true, y_true), (x_pred, y_pred)), distance in distances:
        if ((x_true, y_true) in points_done_true) or (
            (x_pred, y_pred) in points_done_pred
        ):
            # do not consider a taken point again
            continue
        # normalize all distances by diving by the diagonal length
        lowest_dist_pairs.append(
            (
                ((x_true, y_true), (x_pred, y_pred)),
                math.sqrt(float(distance)) / diagonal_length,
            )
        )
        points_done_true.add((x_true, y_true))
        points_done_pred.add((x_pred, y_pred))

    return lowest_dist_pairs


#
# Calc f-score for a point feature set for a single image
#
def calc_f1_score_pts(
    gt_pts,
    pred_pts,
    img_wh,
):
    """
    Results are a list of (feature_label,num_gt_pts,num_pred_pts,precision,recall,f1_score)
    """

    results = []

    for label, xy_pts in gt_pts.items():

        mat_true = np.zeros((img_wh[1], img_wh[0]), dtype=np.uint8)
        for x, y in xy_pts:
            mat_true[int(y), int(x)] = BINARY_PIXEL_VAL

        mat_pred = np.zeros((img_wh[1], img_wh[0]), dtype=np.uint8)
        xy_pts_pred = pred_pts.get(label, [])
        for x, y in xy_pts_pred:
            mat_pred[int(y), int(x)] = BINARY_PIXEL_VAL

        pred_maxval = mat_pred.max()
        if pred_maxval > 1:
            print("WARNING! Scaling prediction raster to be 1s and 0s")
            mat_pred = mat_pred / pred_maxval
        true_maxval = mat_true.max()
        if true_maxval > 1:
            print("WARNING! Scaling gnd truth raster to be 1s and 0s")
            mat_true = mat_true / true_maxval

        num_gt_pts = len(np.argwhere(mat_true > 0))

        num_pred_pts = len(np.argwhere(mat_pred > 0))

        lowest_dist_pairs = overlap_distance_calculate(
            mat_true, mat_pred, min_valid_range=MIN_VALID_RANGE
        )
        # print('len(lowest_dist_pairs):', len(lowest_dist_pairs))
        sum_of_similarities = sum([1.0 - item[1] for item in lowest_dist_pairs])
        count_mat_pred = float(len(np.argwhere(mat_pred == 1)))
        count_mat_true = float(len(np.argwhere(mat_true == 1)))
        precision = (
            sum_of_similarities / count_mat_pred if count_mat_pred > 0.0 else 0.0
        )
        precision = min(precision, 1.0)
        recall = sum_of_similarities / count_mat_true if count_mat_true > 0.0 else 0.0
        recall = min(recall, 1.0)

        # calculate f-score
        if precision + recall != 0:
            f_score = (2 * precision * recall) / (precision + recall)
        else:
            f_score = 0.0

        print(
            f"  Label: {label},  F1: {f_score:.3f},  Num pts: {num_pred_pts} / {num_gt_pts}"
        )

        results.append((label, num_gt_pts, num_pred_pts, precision, recall, f_score))

    return results
