"""
Functionality for computing segmentation metrics for the different structures (surfaces, curves, and blobs) in 3D images
"""

__author__ = 'Antonio Martinez-Sanchez'

import scipy
import numpy as np

from core.diff import diff3d, eig3dk


def s_dice(tomo: np.ndarray, tomo_gt: np.ndarray, thick: float) -> tuple:
    """
    Computes surface DICE metric (s-DICE) for two input segmented tomograms
    :param tomo: input predicted tomogoram (values >0 are considered foregound)
    :param tomo_gt: input ground truth (values >0 are considered foregound)
    :param thick: thickness used to check the overlapping
    :return: returns a 3-tuple where the 1st value is s-DICE, 2nd TP (Topology Precision), and TS
             (Topology Sensitivity)
    """
    assert tomo.shape == tomo_gt.shape
    assert thick >= 0
    tomo_seg = tomo > 0
    tomo_gt_seg = tomo_gt > 0

    # Getting segmentations ridges
    tomo_dsts = scipy.ndimage.morphology.distance_transform_edt(tomo_seg)
    tomo_gt_dsts = scipy.ndimage.morphology.distance_transform_edt(tomo_gt_seg)
    tomo_skel = surface_skel(tomo_dsts, tomo_dsts > 0)
    del tomo_dsts
    tomo_gt_skel = surface_skel(tomo_gt_dsts, tomo_gt_dsts > 0)
    del tomo_gt_dsts

    # Computing the metric
    tp = (tomo_skel * tomo_gt_skel).sum() / tomo_skel.sum()
    ts = (tomo_gt_skel * tomo).sum() / tomo_gt_skel.sum()

    return 2*(tp*ts) / (tp + ts), tp, ts


def surface_skel(tomo: np.ndarray, mask=None) -> np.ndarray:
    """
    From an input tomogram compute its skeleton for surface ridges
    :param tomo: input tomogram
    :param mask: Default None, if given binary mask (np.ndarray) to only consider ridges within the 1-valued voxles
    :return: a binary tomogram with the skeleton
    """
    if mask is None:
        mask = np.ones(shape=tomo.shape, dtype=bool)
    else:
        assert mask.shape == tomo.shape


    # 1st order derivatives
    tomo_x = diff3d(tomo, 0)
    tomo_y = diff3d(tomo, 1)
    tomo_z = diff3d(tomo, 2)

    # Hessian tensor
    tomo_xx = diff3d(tomo_x, 0)
    tomo_yy = diff3d(tomo_y, 1)
    tomo_zz = diff3d(tomo_z, 2)
    tomo_xy = diff3d(tomo_x, 1)
    tomo_xz = diff3d(tomo_x, 2)
    tomo_yz = diff3d(tomo_y, 2)

    # Eigen-problem
    tomo_l1, _, _, tomo_v1x, tomo_v1x, tomo_v1x, _, _, _, _, _, _ = eig3dk(tomo_xx, tomo_yy, tomo_zz,
                                                                           tomo_xy, tomo_xz, tomo_yz)
    # Non-maximum suppression
    return nonmaxsup_surf(tomo_l1, mask, V1x, V1y, V1z)

