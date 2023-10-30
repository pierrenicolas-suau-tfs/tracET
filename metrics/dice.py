"""
Functionality for computing segmentation metrics for the different structures (surfaces, curves, and blobs) in 3D images
"""

__author__ = 'Antonio Martinez-Sanchez'

import scipy
import numpy as np

from core.diff import diff3d, eig3dk, nonmaxsup_surf, nonmaxsup_line, nonmaxsup_point


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
    tomo_l1, _, _, tomo_v1x, tomo_v1y, tomo_v1z, _, _, _, _, _, _ = eig3dk(tomo_xx, tomo_yy, tomo_zz,
                                                                           tomo_xy, tomo_xz, tomo_yz)
    # Non-maximum suppression
    return nonmaxsup_surf(tomo_l1, mask, tomo_v1x, tomo_v1y, tomo_v1z)


def line_skel(tomo: np.ndarray, mask=None, mode='hessian') -> np.ndarray:
    """
    From an input tomogram compute its skeleton for line ridges
    :param tomo: input tomogram
    :param mask: Default None, if given binary mask (np.ndarray) to only consider ridges within the 1-valued voxles
    :param mode: for computing the eigenvalues the Hessian tensor is always used, but for eigenvectors if 'hessian'
                 (default) then the Hessian tensor if 'structure' then the Structure tensor is used
    :return: a binary tomogram with the skeleton
    """
    if mask is None:
        mask = np.ones(shape=tomo.shape, dtype=bool)
    else:
        assert mask.shape == tomo.shape
    assert mode == 'hessian' or mode == 'structure'

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
    tomo_l1, _, _, tomo_v1x, tomo_v1y, tomo_v1z, tomo_v2x, tomo_v2y, tomo_v2z, _, _, _ = eig3dk(tomo_xx, tomo_yy,
                                                                                                tomo_zz, tomo_xy,
                                                                                                tomo_xz, tomo_yz)

    # Structure tensor
    if mode == 'structure':
        del tomo_v1x
        del tomo_v1y
        del tomo_v1z
        del tomo_v2x
        del tomo_v2y
        del tomo_v2z

        tomo_xx = tomo_x * tomo_x
        tomo_yy = tomo_y * tomo_y
        tomo_zz = tomo_z * tomo_z
        tomo_xy = tomo_x * tomo_y
        tomo_xz = tomo_x * tomo_z
        tomo_yz = tomo_y * tomo_z

        _, _, _, tomo_v1x, tomo_v1y, tomo_v1z, tomo_v2x, tomo_v2y, tomo_v2z, _, _, _ = eig3dk(tomo_xx, tomo_yy,
                                                                                              tomo_zz, tomo_xy,
                                                                                              tomo_xz, tomo_yz)

    # Non-maximum suppression
    return nonmaxsup_line(tomo_l1, mask, tomo_v1x, tomo_v1y, tomo_v1z, tomo_v2x, tomo_v2y, tomo_v2z)


def point_skel(tomo: np.ndarray, mask=None, mode='hessian') -> np.ndarray:
    """
    From an input tomogram compute its skeleton for pint ridges
    :param tomo: input tomogram
    :param mask: Default None, if given binary mask (np.ndarray) to only consider ridges within the 1-valued voxles
    :param mode: for computing the eigenvalues the Hessian tensor is always used, but for eigenvectors if 'hessian'
                 (default) then the Hessian tensor if 'structure' then the Structure tensor is used
    :return: a binary tomogram with the skeleton
    """
    if mask is None:
        mask = np.ones(shape=tomo.shape, dtype=bool)
    else:
        assert mask.shape == tomo.shape
    assert mode == 'hessian' or mode == 'structure'

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
    (tomo_l1, _, _,
     tomo_v1x, tomo_v1y, tomo_v1z,
     tomo_v2x, tomo_v2y, tomo_v2z,
     tomo_v3x, tomo_v3y, tomo_v3z) = eig3dk(tomo_xx, tomo_yy, tomo_zz, tomo_xy, tomo_xz, tomo_yz)

    # Structure tensor
    if mode == 'structure':
        del tomo_v1x
        del tomo_v1y
        del tomo_v1z
        del tomo_v2x
        del tomo_v2y
        del tomo_v2z
        del tomo_v3x
        del tomo_v3y
        del tomo_v3z

        tomo_xx = tomo_x * tomo_x
        tomo_yy = tomo_y * tomo_y
        tomo_zz = tomo_z * tomo_z
        tomo_xy = tomo_x * tomo_y
        tomo_xz = tomo_x * tomo_z
        tomo_yz = tomo_y * tomo_z

        (_, _, _,
         tomo_v1x, tomo_v1y, tomo_v1z,
         tomo_v2x, tomo_v2y, tomo_v2z,
         tomo_v3x, tomo_v3y, tomo_v3z) = eig3dk(tomo_xx, tomo_yy, tomo_zz, tomo_xy, tomo_xz, tomo_yz)

    # Non-maximum suppression
    return nonmaxsup_point(tomo_l1, mask, tomo_v1x, tomo_v1y, tomo_v1z, tomo_v2x, tomo_v2y, tomo_v2z,
                           tomo_v3x, tomo_v3y, tomo_v3z)


def cs_dice(tomo: np.ndarray, tomo_gt: np.ndarray, skel=None, skel_gt=None) -> tuple:
    """
    Computes surface DICE metric (s-DICE) for two input segmented tomograms
    :param tomo: input predicted tomogoram (values >0 are considered foregound)
    :param tomo_gt: input ground truth (values >0 are considered foregound)
    :param skel: default None, if a np.ndarray with the same shape as tomogram is giving it is filled with the skeleton
                 generated for computing the metric
    :param skel_gt: default None, if a np.ndarray with the same shape as ground truth tomogram is giving it is filled
                    with the skeleton generated for computing the metric
    :return: returns a 3-tuple where the 1st value is cs-DICE, 2nd TP (Topology Precision), and TS
             (Topology Sensitivity)
    """
    assert tomo.shape == tomo_gt.shape
    if skel is not None:
        assert isinstance(skel, np.ndarray)
        assert skel.shape == tomo.shape
    if skel_gt is not None:
        assert isinstance(skel_gt, np.ndarray)
        assert skel_gt.shape == tomo_gt.shape
    tomo_seg = tomo > 0
    tomo_gt_seg = tomo_gt > 0

    # Getting segmentations ridges
    tomo_dsts = scipy.ndimage.morphology.distance_transform_edt(tomo_seg).astype(np.float32)
    tomo_gt_dsts = scipy.ndimage.morphology.distance_transform_edt(tomo_gt_seg).astype(np.float32)
    tomo_skel = surface_skel(tomo_dsts, tomo_dsts > 0)
    if skel is not None:
        skel = tomo_skel
    del tomo_dsts
    tomo_gt_skel = surface_skel(tomo_gt_dsts, tomo_gt_dsts > 0)
    if skel_gt is not None:
        skel_gt = tomo_gt_skel
    del tomo_gt_dsts

    # Computing the metric
    tp = (tomo_skel * tomo_gt_skel).sum() / tomo_skel.sum()
    ts = (tomo_gt_skel * tomo).sum() / tomo_gt_skel.sum()

    return 2*(tp*ts) / (tp + ts), tp, ts


def cl_dice(tomo: np.ndarray, tomo_gt: np.ndarray, skel=None, skel_gt=None) -> tuple:
    """
    Computes centerline DICE metric (cl-DICE) for two input segmented tomograms
    :param tomo: input predicted tomogoram (values >0 are considered foregound)
    :param tomo_gt: input ground truth (values >0 are considered foregound)
    :param skel: default None, if a np.ndarray with the same shape as tomogram is giving it is filled with the skeleton
                 generated for computing the metric
    :param skel_gt: default None, if a np.ndarray with the same shape as ground truth tomogram is giving it is filled
                    with the skeleton generated for computing the metric
    :return: returns a 3-tuple where the 1st value is cl-DICE, 2nd TP (Topology Precision), and TS
             (Topology Sensitivity)
    """
    assert tomo.shape == tomo_gt.shape
    if skel is not None:
        assert isinstance(skel, np.ndarray)
        assert skel.shape == tomo.shape
    if skel_gt is not None:
        assert isinstance(skel_gt, np.ndarray)
        assert skel_gt.shape == tomo_gt.shape
    tomo_seg = tomo > 0
    tomo_gt_seg = tomo_gt > 0

    # Getting segmentations ridges
    tomo_dsts = scipy.ndimage.morphology.distance_transform_edt(tomo_seg)
    tomo_gt_dsts = scipy.ndimage.morphology.distance_transform_edt(tomo_gt_seg)
    tomo_skel = line_skel(tomo_dsts, tomo_dsts > 0)
    if skel is not None:
        skel = tomo_skel
    del tomo_dsts
    tomo_gt_skel = line_skel(tomo_gt_dsts, tomo_gt_dsts > 0)
    if skel_gt is not None:
        skel_gt = tomo_gt_skel
    del tomo_gt_dsts

    # Computing the metric
    tp = (tomo_skel * tomo_gt_skel).sum() / tomo_skel.sum()
    ts = (tomo_gt_skel * tomo).sum() / tomo_gt_skel.sum()

    return 2*(tp*ts) / (tp + ts), tp, ts


def pt_dice(tomo: np.ndarray, tomo_gt: np.ndarray, skel=None, skel_gt=None) -> tuple:
    """
    Computes point DICE metric (pt-DICE) for two input segmented tomograms
    :param tomo: input predicted tomogoram (values >0 are considered foregound)
    :param tomo_gt: input ground truth (values >0 are considered foregound)
    :param skel: default None, if a np.ndarray with the same shape as tomogram is giving it is filled with the skeleton
                 generated for computing the metric
    :param skel_gt: default None, if a np.ndarray with the same shape as ground truth tomogram is giving it is filled
                    with the skeleton generated for computing the metric
    :return: returns a 3-tuple where the 1st value is pt-DICE, 2nd TP (Topology Precision), and TS
             (Topology Sensitivity)
    """
    assert tomo.shape == tomo_gt.shape
    if skel is not None:
        assert isinstance(skel, np.ndarray)
        assert skel.shape == tomo.shape
    if skel_gt is not None:
        assert isinstance(skel_gt, np.ndarray)
        assert skel_gt.shape == tomo_gt.shape
    tomo_seg = tomo > 0
    tomo_gt_seg = tomo_gt > 0

    # Getting segmentations ridges
    tomo_dsts = scipy.ndimage.morphology.distance_transform_edt(tomo_seg)
    tomo_gt_dsts = scipy.ndimage.morphology.distance_transform_edt(tomo_gt_seg)
    tomo_skel = point_skel(tomo_dsts, tomo_dsts > 0)
    if skel is not None:
        skel = tomo_skel
    del tomo_dsts
    tomo_gt_skel = point_skel(tomo_gt_dsts, tomo_gt_dsts > 0)
    if skel_gt is not None:
        skel_gt = tomo_gt_skel
    del tomo_gt_dsts

    # Computing the metric
    tp = (tomo_skel * tomo_gt_skel).sum() / tomo_skel.sum()
    ts = (tomo_gt_skel * tomo).sum() / tomo_gt_skel.sum()

    return 2*(tp*ts) / (tp + ts), tp, ts


