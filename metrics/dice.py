"""
Functionality for computing segmentation metrics for the different structures (surfaces, curves, and blobs) in 3D images
"""

__author__ = 'Antonio Martinez-Sanchez'

import scipy
import numpy as np
from core import lio
from core.diff import diff3d, nonmaxsup_surf, nonmaxsup_line, nonmaxsup_point, angauss
from supression import desyevv, nonmaxsup_2
from mt.representation import points_to_btomo, seg_dist_trans

def thick_dst(map,zf=1,mt_dst=15):
    tomo_dst = seg_dist_trans(map)
    tomo_mt = tomo_dst <= mt_dst * zf
    tomo_mt_dst = tomo_dst - mt_dst
    tomo_mt_dst[tomo_mt_dst > 0] = 0
    tomo_mt_dst *= -1
    return tomo_mt_dst

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
    tomo_x = diff3d(tomo, 0).astype(np.float32)
    tomo_y = diff3d(tomo, 1).astype(np.float32)
    tomo_z = diff3d(tomo, 2).astype(np.float32)

    # Hessian tensor
    tomo_xx = np.swapaxes(diff3d(tomo_x, 0),0,2).flatten()
    tomo_yy = np.swapaxes(diff3d(tomo_y, 1),0,2).flatten()
    tomo_zz = np.swapaxes(diff3d(tomo_z, 2),0,2).flatten()
    tomo_xy = np.swapaxes(diff3d(tomo_x, 1),0,2).flatten()
    tomo_xz = np.swapaxes(diff3d(tomo_x, 2),0,2).flatten()
    tomo_yz = np.swapaxes(diff3d(tomo_y, 2),0,2).flatten()
    del tomo_x
    del tomo_y
    del tomo_z

    # C-processing eigen-problem
    tomo_l, _, _, tomo_v1x, tomo_v1y, tomo_v1z, _, _, _, _, _, _ = desyevv(tomo_xx, tomo_yy, tomo_zz,
                                                                            tomo_xy, tomo_xz, tomo_yz)
    del tomo_xx
    del tomo_yy
    del tomo_zz
    del tomo_xy
    del tomo_xz
    del tomo_yz

    # Non-maximum suppression
    [Nx, Ny, Nz] = np.shape(tomo)
    mask_h = np.zeros((Nx, Ny, Nz))
    mask_h[1:Nx - 2, 1:Ny - 2, 1:Nz - 2] = 1
    mask = mask * mask_h
    del mask_h
    mask_h = np.swapaxes(mask > 0, 0, 2).flatten().astype(bool)
    mask_ids = np.arange(0, Nx * Ny * Nz, dtype=np.int64)
    mask_ids = mask_ids[mask_h]
    del mask_h
    #tomo_l = np.swapaxes(tomo_l.astype(np.float32), 0, 2).flatten()
    dim = np.array([Nx, Ny]).astype('uint32')
    supred = np.swapaxes(np.reshape(nonmaxsup_2(tomo_l, tomo_v1x, tomo_v1y, tomo_v1z, mask_ids, dim),(Nz, Ny, Nx)), 0, 2)
    tomo_l = np.swapaxes(np.reshape(tomo_l, (Nz, Ny, Nx)), 0, 2)
    tomo_v1x = np.swapaxes(np.reshape(tomo_v1x, (Nz, Ny, Nx)), 0, 2)
    tomo_v1y = np.swapaxes(np.reshape(tomo_v1y, (Nz, Ny, Nx)), 0, 2)
    tomo_v1z = np.swapaxes(np.reshape(tomo_v1z, (Nz, Ny, Nx)), 0, 2)

    return supred, tomo_l, tomo_v1x, tomo_v1y, tomo_v1z


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
    [Nx, Ny, Nz] = np.shape(tomo)
    tomo_x = diff3d(tomo, 0).astype(np.float32)
    tomo_y = diff3d(tomo, 1).astype(np.float32)
    tomo_z = diff3d(tomo, 2).astype(np.float32)

    # Hessian tensor
    tomo_xx = np.swapaxes(diff3d(tomo_x, 0), 0, 2).flatten()
    tomo_yy = np.swapaxes(diff3d(tomo_y, 1), 0, 2).flatten()
    tomo_zz = np.swapaxes(diff3d(tomo_z, 2), 0, 2).flatten()
    tomo_xy = np.swapaxes(diff3d(tomo_x, 1), 0, 2).flatten()
    tomo_xz = np.swapaxes(diff3d(tomo_x, 2), 0, 2).flatten()
    tomo_yz = np.swapaxes(diff3d(tomo_y, 2), 0, 2).flatten()
    if mode != 'structure':
        del tomo_x
        del tomo_y
        del tomo_z

    # C-processing eigen-problem
    tomo_l1, _, _, tomo_v1x, tomo_v1y, tomo_v1z, tomo_v2x, tomo_v2y, tomo_v2z, _, _, _ = desyevv(tomo_xx, tomo_yy,
                                                                                                 tomo_zz, tomo_xy,
                                                                                                 tomo_xz, tomo_yz)

    if mode != 'structure':
        tomo_l1 = np.swapaxes(np.reshape(np.abs(tomo_l1), (Nz, Ny, Nx)), 0, 2)
        tomo_v1x = np.swapaxes(np.reshape(tomo_v1x, (Nz, Ny, Nx)), 0, 2)
        tomo_v1y = np.swapaxes(np.reshape(tomo_v1y, (Nz, Ny, Nx)), 0, 2)
        tomo_v1z = np.swapaxes(np.reshape(tomo_v1z, (Nz, Ny, Nx)), 0, 2)
        tomo_v2x = np.swapaxes(np.reshape(tomo_v2x, (Nz, Ny, Nx)), 0, 2)
        tomo_v2y = np.swapaxes(np.reshape(tomo_v2y, (Nz, Ny, Nx)), 0, 2)
        tomo_v2z = np.swapaxes(np.reshape(tomo_v2z, (Nz, Ny, Nx)), 0, 2)

    # Structure tensor
    if mode == 'structure':
        del tomo_v1x
        del tomo_v1y
        del tomo_v1z
        del tomo_v2x
        del tomo_v2y
        del tomo_v2z

        tomo_xx = (tomo_x * tomo_x).flatten()
        tomo_yy = (tomo_y * tomo_y).flatten()
        tomo_zz = (tomo_z * tomo_z).flatten()
        tomo_xy = (tomo_x * tomo_y).flatten()
        tomo_xz = (tomo_x * tomo_z).flatten()
        tomo_yz = (tomo_y * tomo_z).flatten()

        # C-processing eigen-problem
        _, _, _, tomo_v1x, tomo_v1y, tomo_v1z, tomo_v2x, tomo_v2y, tomo_v2z, _, _, _ = desyevv(tomo_xx, tomo_yy,
                                                                                                     tomo_zz, tomo_xy,
                                                                                                     tomo_xz, tomo_yz)
        tomo_v1x = np.swapaxes(np.reshape(tomo_v1x, (Nz, Ny, Nx)), 0, 2)
        tomo_v1y = np.swapaxes(np.reshape(tomo_v1y, (Nz, Ny, Nx)), 0, 2)
        tomo_v1z = np.swapaxes(np.reshape(tomo_v1z, (Nz, Ny, Nx)), 0, 2)
        tomo_v2x = np.swapaxes(np.reshape(tomo_v2x, (Nz, Ny, Nx)), 0, 2)
        tomo_v2y = np.swapaxes(np.reshape(tomo_v2y, (Nz, Ny, Nx)), 0, 2)
        tomo_v2z = np.swapaxes(np.reshape(tomo_v2z, (Nz, Ny, Nx)), 0, 2)

    # Non-maximum suppression
    return nonmaxsup_line(tomo_l1, mask, tomo_v1x, tomo_v1y, tomo_v1z, tomo_v2x, tomo_v2y, tomo_v2z), tomo_l1, tomo_v1x, tomo_v1y, tomo_v1z, tomo_v2x, tomo_v2y, tomo_v2z


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
    [Nx, Ny, Nz] = np.shape(tomo)
    tomo_x = diff3d(tomo, 0).astype(np.float32)
    tomo_y = diff3d(tomo, 1).astype(np.float32)
    tomo_z = diff3d(tomo, 2).astype(np.float32)

    # Hessian tensor
    tomo_xx = diff3d(tomo_x, 0).flatten()
    tomo_yy = diff3d(tomo_y, 1).flatten()
    tomo_zz = diff3d(tomo_z, 2).flatten()
    tomo_xy = diff3d(tomo_x, 1).flatten()
    tomo_xz = diff3d(tomo_x, 2).flatten()
    tomo_yz = diff3d(tomo_y, 2).flatten()
    if mode != 'structure':
        del tomo_x
        del tomo_y
        del tomo_z

    # C-processing eigen-problem
    (tomo_l1, _, _,
     tomo_v1x, tomo_v1y, tomo_v1z,
     tomo_v2x, tomo_v2y, tomo_v2z,
     tomo_v3x, tomo_v3y, tomo_v3z) = desyevv(tomo_xx, tomo_yy, tomo_zz, tomo_xy, tomo_xz, tomo_yz)
    if mode != 'structure':
        tomo_l1 = np.swapaxes(np.reshape(tomo_l1, (Nz, Ny, Nx)), 0, 2)
        tomo_v1x = np.swapaxes(np.reshape(tomo_v1x, (Nz, Ny, Nx)), 0, 2)
        tomo_v1y = np.swapaxes(np.reshape(tomo_v1y, (Nz, Ny, Nx)), 0, 2)
        tomo_v1z = np.swapaxes(np.reshape(tomo_v1z, (Nz, Ny, Nx)), 0, 2)
        tomo_v2x = np.swapaxes(np.reshape(tomo_v2x, (Nz, Ny, Nx)), 0, 2)
        tomo_v2y = np.swapaxes(np.reshape(tomo_v2y, (Nz, Ny, Nx)), 0, 2)
        tomo_v2z = np.swapaxes(np.reshape(tomo_v2z, (Nz, Ny, Nx)), 0, 2)
        tomo_v3x = np.swapaxes(np.reshape(tomo_v3x, (Nz, Ny, Nx)), 0, 2)
        tomo_v3y = np.swapaxes(np.reshape(tomo_v3y, (Nz, Ny, Nx)), 0, 2)
        tomo_v3z = np.swapaxes(np.reshape(tomo_v3z, (Nz, Ny, Nx)), 0, 2)

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

        tomo_xx = (tomo_x * tomo_x).flatten()
        tomo_yy = (tomo_y * tomo_y).flatten()
        tomo_zz = (tomo_z * tomo_z).flatten()
        tomo_xy = (tomo_x * tomo_y).flatten()
        tomo_xz = (tomo_x * tomo_z).flatten()
        tomo_yz = (tomo_y * tomo_z).flatten()

        (_, _, _,
         tomo_v1x, tomo_v1y, tomo_v1z,
         tomo_v2x, tomo_v2y, tomo_v2z,
         tomo_v3x, tomo_v3y, tomo_v3z) = desyevv(tomo_xx, tomo_yy, tomo_zz, tomo_xy, tomo_xz, tomo_yz)
        tomo_v1x = np.swapaxes(np.reshape(tomo_v1x, (Nz, Ny, Nx)), 0, 2)
        tomo_v1y = np.swapaxes(np.reshape(tomo_v1y, (Nz, Ny, Nx)), 0, 2)
        tomo_v1z = np.swapaxes(np.reshape(tomo_v1z, (Nz, Ny, Nx)), 0, 2)
        tomo_v2x = np.swapaxes(np.reshape(tomo_v2x, (Nz, Ny, Nx)), 0, 2)
        tomo_v2y = np.swapaxes(np.reshape(tomo_v2y, (Nz, Ny, Nx)), 0, 2)
        tomo_v2z = np.swapaxes(np.reshape(tomo_v2z, (Nz, Ny, Nx)), 0, 2)
        tomo_v3x = np.swapaxes(np.reshape(tomo_v3x, (Nz, Ny, Nx)), 0, 2)
        tomo_v3y = np.swapaxes(np.reshape(tomo_v3y, (Nz, Ny, Nx)), 0, 2)
        tomo_v3z = np.swapaxes(np.reshape(tomo_v3z, (Nz, Ny, Nx)), 0, 2)

    # Non-maximum suppression
    return nonmaxsup_point(tomo_l1, mask, tomo_v1x, tomo_v1y, tomo_v1z, tomo_v2x, tomo_v2y, tomo_v2z,
                           tomo_v3x, tomo_v3y, tomo_v3z)


def cs_dice(tomo: np.ndarray, tomo_gt: np.ndarray, dilation=0) -> tuple:
    """
    Computes surface DICE metric (s-DICE) for two input segmented tomograms
    :param tomo: input predicted tomogoram (values >0 are considered foreground)
    :param tomo_gt: input ground truth (values >0 are considered foreground)
    :param dilation: number of iterations to dilate the segmentation (default 0)
    :return: returns a 5-tuple where the 1st value is cs-DICE, 2nd TP (Topology Precision), and TS
             (Topology Sensitivity), tomogram skeleton, ground truth skeleton
    """
    assert tomo.shape == tomo_gt.shape

    # Getting segmentations ridges
    tomo_dsts = scipy.ndimage.morphology.distance_transform_edt(tomo > 0).astype(np.float32)
    tomo_skel, tomo_l1, tomo_v1x, tomo_v1y, tomo_v1z = surface_skel(tomo_dsts, tomo_dsts > 0)
    #lio.write_mrc(tomo_l1.astype(np.float32),
                  #'/project/chiem/pelayo/neural_network/try_dice/desyevv_sur/Ctrl_20220511_368d_tomo06_seg_L1.mrc')
    #lio.write_mrc(tomo_v1x.astype(np.float32),
                  #'/project/chiem/pelayo/neural_network/try_dice/desyevv_sur/Ctrl_20220511_368d_tomo06_seg_V1x.mrc')
    #lio.write_mrc(tomo_v1y.astype(np.float32),
                  #'/project/chiem/pelayo/neural_network/try_dice/desyevv_sur/Ctrl_20220511_368d_tomo06_seg_V1y.mrc')
    #lio.write_mrc(tomo_v1z.astype(np.float32),
                  #'/project/chiem/pelayo/neural_network/try_dice/desyevv_sur/Ctrl_20220511_368d_tomo06_seg_V1z.mrc')
    del tomo_dsts
    tomo_gt_dsts = scipy.ndimage.morphology.distance_transform_edt(tomo_gt > 0).astype(np.float32)
    tomo_gt_skel, tomo_l1, tomo_v1x, tomo_v1y, tomo_v1z = surface_skel(tomo_gt_dsts, tomo_gt_dsts > 0)
    #lio.write_mrc(tomo_l1.astype(np.float32),
    #              '/project/chiem/pelayo/neural_network/try_dice/desyevv_sur/Ctrl_20220511_368d_tomo06_gt_L1.mrc')
    #lio.write_mrc(tomo_v1x.astype(np.float32),
     #             '/project/chiem/pelayo/neural_network/try_dice/desyevv_sur/Ctrl_20220511_368d_tomo06_gt_V1x.mrc')
    #lio.write_mrc(tomo_v1y.astype(np.float32),
     #             '/project/chiem/pelayo/neural_network/try_dice/desyevv_sur/Ctrl_20220511_368d_tomo06_gt_V1y.mrc')
    #lio.write_mrc(tomo_v1z.astype(np.float32),
     #             '/project/chiem/pelayo/neural_network/try_dice/desyevv_sur/Ctrl_20220511_368d_tomo06_gt_V1z.mrc')
    del tomo_gt_dsts

    # Dilation
    if dilation > 0:
        tomo_d = scipy.ndimage.binary_dilation(tomo, iterations=dilation)
        tomo_gt_d = scipy.ndimage.binary_dilation(tomo_gt, iterations=dilation)
    else:
        tomo_d = tomo
        tomo_gt_d = tomo_gt

    # Computing the metric
    tp = (tomo_skel * tomo_gt_d).sum() / tomo_skel.sum()
    ts = (tomo_gt_skel * tomo_d).sum() / tomo_gt_skel.sum()

    return 2*(tp*ts) / (tp + ts), tp, ts, tomo_skel, tomo_gt_skel


def cl_dice(tomo: np.ndarray, tomo_gt: np.ndarray, dilation=0) -> tuple:
    """
    Computes centerline DICE metric (cl-DICE) for two input segmented tomograms
    :param tomo: input predicted tomogoram (values >0 are considered foreground)
    :param tomo_gt: input ground truth (values >0 are considered foreground)
    :param dilation: number of iterations to dilate the segmentation (default 0)
    :return: returns a 3-tuple where the 1st value is cl-DICE, 2nd TP (Topology Precision), and TS
             (Topology Sensitivity)
    """
    assert tomo.shape == tomo_gt.shape
    tomo_seg = tomo > 0
    tomo_gt_seg = tomo_gt > 0


    # Getting segmentations ridges
    tomo_dsts = scipy.ndimage.morphology.distance_transform_edt(tomo_seg)
    tomo_gt_dsts = scipy.ndimage.morphology.distance_transform_edt(tomo_gt_seg)
    #tomo_dsts=thick_dst(tomo_seg)
    #tomo_gt_dsts=thick_dst(tomo_gt_seg)
    #lio.write_mrc(tomo_dsts.astype(np.float32),'/project/chiem/pelayo/neural_network/try_dice/distance_transf/Ctrl_20220511_368d_tomo06_seg_dst.mrc')
    #lio.write_mrc(tomo_gt_dsts.astype(np.float32),
                  #'/project/chiem/pelayo/neural_network/try_dice/distance_transf/Ctrl_20220511_368d_tomo06_gt_dst.mrc')
    tomo_skel, tomo_l1, tomo_v1x, tomo_v1y, tomo_v1z, tomo_v2x, tomo_v2y, tomo_v2z = line_skel(tomo_dsts, tomo_dsts > 0)
    #lio.write_mrc(tomo_l1.astype(np.float32),
                  #'/project/chiem/pelayo/neural_network/try_dice/desyevv/Ctrl_20220511_368d_tomo06_seg_L1.mrc')
    #lio.write_mrc(tomo_v1x.astype(np.float32),
                  #'/project/chiem/pelayo/neural_network/try_dice/desyevv/Ctrl_20220511_368d_tomo06_seg_V1x.mrc')
    #lio.write_mrc(tomo_v1y.astype(np.float32),
                  #'/project/chiem/pelayo/neural_network/try_dice/desyevv/Ctrl_20220511_368d_tomo06_seg_V1y.mrc')
    #lio.write_mrc(tomo_v1z.astype(np.float32),
                  #'/project/chiem/pelayo/neural_network/try_dice/desyevv/Ctrl_20220511_368d_tomo06_seg_V1z.mrc')
    #lio.write_mrc(tomo_v2x.astype(np.float32),
                  #'/project/chiem/pelayo/neural_network/try_dice/desyevv/Ctrl_20220511_368d_tomo06_seg_V2x.mrc')
    #lio.write_mrc(tomo_v2y.astype(np.float32),
                  #'/project/chiem/pelayo/neural_network/try_dice/desyevv/Ctrl_20220511_368d_tomo06_seg_V2y.mrc')
    #lio.write_mrc(tomo_v2z.astype(np.float32),
                  #'/project/chiem/pelayo/neural_network/try_dice/desyevv/Ctrl_20220511_368d_tomo06_seg_V2z.mrc')
    del tomo_dsts
    tomo_gt_skel, tomo_l1, tomo_v1x, tomo_v1y, tomo_v1z, tomo_v2x, tomo_v2y, tomo_v2z = line_skel(tomo_gt_dsts, tomo_gt_dsts > 0)
    lio.write_mrc(tomo_l1.astype(np.float32),
                  '/project/chiem/pelayo/neural_network/try_dice/desyevv/Ctrl_20220511_368d_tomo06_gt_L1.mrc')
    lio.write_mrc(tomo_v1x.astype(np.float32),
                  '/project/chiem/pelayo/neural_network/try_dice/desyevv/Ctrl_20220511_368d_tomo06_gt_V1x.mrc')
    lio.write_mrc(tomo_v1y.astype(np.float32),
                  '/project/chiem/pelayo/neural_network/try_dice/desyevv/Ctrl_20220511_368d_tomo06_gt_V1y.mrc')
    lio.write_mrc(tomo_v1z.astype(np.float32),
                  '/project/chiem/pelayo/neural_network/try_dice/desyevv/Ctrl_20220511_368d_tomo06_gt_V1z.mrc')
    lio.write_mrc(tomo_v2x.astype(np.float32),
                  '/project/chiem/pelayo/neural_network/try_dice/desyevv/Ctrl_20220511_368d_tomo06_gt_V2x.mrc')
    lio.write_mrc(tomo_v2y.astype(np.float32),
                  '/project/chiem/pelayo/neural_network/try_dice/desyevv/Ctrl_20220511_368d_tomo06_gt_V2y.mrc')
    lio.write_mrc(tomo_v2z.astype(np.float32),
                  '/project/chiem/pelayo/neural_network/try_dice/desyevv/Ctrl_20220511_368d_tomo06_gt_V2z.mrc')
    del tomo_gt_dsts

    # Dilation
    if dilation > 0:
        tomo_d = scipy.ndimage.binary_dilation(tomo, iterations=dilation)
        tomo_gt_d = scipy.ndimage.binary_dilation(tomo_gt, iterations=dilation)
    else:
        tomo_d = tomo
        tomo_gt_d = tomo_gt

    # Computing the metric
    tp = (tomo_skel * tomo_gt_d).sum() / tomo_skel.sum()
    ts = (tomo_gt_skel * tomo_d).sum() / tomo_gt_skel.sum()

    return 2*(tp*ts) / (tp + ts), tp, ts, tomo_skel, tomo_gt_skel

def cl_dice_soft(tomo: np.ndarray, tomo_gt: np.ndarray, dilation=0, tomo_bin=True, tomo_gt_bin=True, inf=None, tf =None) -> tuple:
    """
        Computes centerline DICE metric (cl-DICE) for two input segmented tomograms
        :param tomo: input predicted tomogoram (values >0 are considered foreground)
        :param tomo_gt: input ground truth (values >0 are considered foreground)
        :param dilation: number of iterations to dilate the segmentation (default 0)
        :return: returns a 3-tuple where the 1st value is cl-DICE, 2nd TP (Topology Precision), and TS
                 (Topology Sensitivity)
        """
    assert tomo.shape == tomo_gt.shape
    if tomo_bin:
        tomo_seg = tomo > 0
        tomo_dsts = scipy.ndimage.morphology.distance_transform_edt(tomo_seg)
        tomo_dsts = angauss(tomo_dsts,3)
    else:
        if inf == None:
            mask = np.ones_like(tomo)
        else:
            mask = np.zeros_like(tomo)
            mask[tomo > inf] = 1
        tomo_dsts = tomo * mask

    if tomo_gt_bin:
        tomo_gt_seg = tomo_gt > 0
        tomo_gt_dsts = scipy.ndimage.morphology.distance_transform_edt(tomo_gt_seg)
        tomo_gt_dsts = angauss(tomo_gt_dsts,6)
    else:
        if tf == None:
            mask_gt = np.ones_like(tomo_gt)
        else:
            mask_gt = np.zeros_like(tomo_gt)
            mask_gt[tomo_gt > tf] = 1
        tomo_gt_dsts = tomo_gt * mask_gt


    # Getting segmentations ridges


    # tomo_dsts=thick_dst(tomo_seg)
    # tomo_gt_dsts=thick_dst(tomo_gt_seg)
    lio.write_mrc(tomo_dsts.astype(np.float32),'/project/chiem/pelayo/neural_network/try_dice/distance_transf/Ctrl_20220511_368d_tomo06_seg_dst.mrc')
    lio.write_mrc(tomo_gt_dsts.astype(np.float32),
    '/project/chiem/pelayo/neural_network/try_dice/distance_transf/Ctrl_20220511_368d_tomo06_gt_dst.mrc')
    tomo_skel, tomo_l1, tomo_v1x, tomo_v1y, tomo_v1z, tomo_v2x, tomo_v2y, tomo_v2z = line_skel(tomo_dsts, tomo_dsts > 0)
    # lio.write_mrc(tomo_l1.astype(np.float32),
    # '/project/chiem/pelayo/neural_network/try_dice/desyevv/Ctrl_20220511_368d_tomo06_seg_L1.mrc')
    # lio.write_mrc(tomo_v1x.astype(np.float32),
    # '/project/chiem/pelayo/neural_network/try_dice/desyevv/Ctrl_20220511_368d_tomo06_seg_V1x.mrc')
    # lio.write_mrc(tomo_v1y.astype(np.float32),
    # '/project/chiem/pelayo/neural_network/try_dice/desyevv/Ctrl_20220511_368d_tomo06_seg_V1y.mrc')
    # lio.write_mrc(tomo_v1z.astype(np.float32),
    # '/project/chiem/pelayo/neural_network/try_dice/desyevv/Ctrl_20220511_368d_tomo06_seg_V1z.mrc')
    # lio.write_mrc(tomo_v2x.astype(np.float32),
    # '/project/chiem/pelayo/neural_network/try_dice/desyevv/Ctrl_20220511_368d_tomo06_seg_V2x.mrc')
    # lio.write_mrc(tomo_v2y.astype(np.float32),
    # '/project/chiem/pelayo/neural_network/try_dice/desyevv/Ctrl_20220511_368d_tomo06_seg_V2y.mrc')
    # lio.write_mrc(tomo_v2z.astype(np.float32),
    # '/project/chiem/pelayo/neural_network/try_dice/desyevv/Ctrl_20220511_368d_tomo06_seg_V2z.mrc')
    del tomo_dsts
    tomo_gt_skel, tomo_l1, tomo_v1x, tomo_v1y, tomo_v1z, tomo_v2x, tomo_v2y, tomo_v2z = line_skel(tomo_gt_dsts,
                                                                                                  tomo_gt_dsts > 0)
    lio.write_mrc(tomo_l1.astype(np.float32),
                  '/project/chiem/pelayo/neural_network/try_dice/desyevv/Ctrl_20220511_368d_tomo06_gt_L1.mrc')
    lio.write_mrc(tomo_v1x.astype(np.float32),
                  '/project/chiem/pelayo/neural_network/try_dice/desyevv/Ctrl_20220511_368d_tomo06_gt_V1x.mrc')
    lio.write_mrc(tomo_v1y.astype(np.float32),
                  '/project/chiem/pelayo/neural_network/try_dice/desyevv/Ctrl_20220511_368d_tomo06_gt_V1y.mrc')
    lio.write_mrc(tomo_v1z.astype(np.float32),
                  '/project/chiem/pelayo/neural_network/try_dice/desyevv/Ctrl_20220511_368d_tomo06_gt_V1z.mrc')
    lio.write_mrc(tomo_v2x.astype(np.float32),
                  '/project/chiem/pelayo/neural_network/try_dice/desyevv/Ctrl_20220511_368d_tomo06_gt_V2x.mrc')
    lio.write_mrc(tomo_v2y.astype(np.float32),
                  '/project/chiem/pelayo/neural_network/try_dice/desyevv/Ctrl_20220511_368d_tomo06_gt_V2y.mrc')
    lio.write_mrc(tomo_v2z.astype(np.float32),
                  '/project/chiem/pelayo/neural_network/try_dice/desyevv/Ctrl_20220511_368d_tomo06_gt_V2z.mrc')
    del tomo_gt_dsts

    # Dilation
    if tomo_bin:
        tomo_d = tomo
    else:
        tomo_d = mask
    if tomo_gt_bin:
        tomo_gt_d= tomo_gt
    else:
        tomo_gt_d=mask_gt

    if dilation > 0:
        tomo_d = scipy.ndimage.binary_dilation(tomo_d, iterations=dilation)
        tomo_gt_d = scipy.ndimage.binary_dilation(tomo_gt_d, iterations=dilation)


    # Computing the metric
    tp = (tomo_skel * tomo_gt_d).sum() / tomo_skel.sum()
    lio.write_mrc((tomo_skel * tomo_gt_d).astype(np.float32),
                  '/project/chiem/pelayo/neural_network/try_dice/prov_map_skel/Ctrl_20220511_368d_tomo06_det_skelxgt.mrc')
    ts = (tomo_gt_skel * tomo_d).sum() / tomo_gt_skel.sum()

    return 2 * (tp * ts) / (tp + ts), tp, ts, tomo_skel, tomo_gt_skel

def pt_dice(tomo: np.ndarray, tomo_gt: np.ndarray, dilation=0) -> tuple:
    """
    Computes point DICE metric (pt-DICE) for two input segmented tomograms
    :param tomo: input predicted tomogoram (values >0 are considered foreground)
    :param tomo_gt: input ground truth (values >0 are considered foreground)
    :param dilation: number of iterations to dilate the segmentation (default 0)
    :return: returns a 3-tuple where the 1st value is pt-DICE, 2nd TP (Topology Precision), and TS
             (Topology Sensitivity)
    """
    assert tomo.shape == tomo_gt.shape
    tomo_seg = tomo > 0
    tomo_gt_seg = tomo_gt > 0

    # Getting segmentations ridges
    tomo_dsts = scipy.ndimage.morphology.distance_transform_edt(tomo_seg)
    tomo_gt_dsts = scipy.ndimage.morphology.distance_transform_edt(tomo_gt_seg)
    tomo_skel = point_skel(tomo_dsts, tomo_dsts > 0)
    del tomo_dsts
    tomo_gt_skel = point_skel(tomo_gt_dsts, tomo_gt_dsts > 0)
    del tomo_gt_dsts

    # Dilation
    if dilation > 0:
        tomo_d = scipy.ndimage.binary_dilation(tomo, iterations=dilation)
        tomo_gt_d = scipy.ndimage.binary_dilation(tomo_gt, iterations=dilation)
    else:
        tomo_d = tomo
        tomo_gt_d = tomo_gt

    # Computing the metric
    tp = (tomo_skel * tomo_gt_d).sum() / tomo_skel.sum()

    ts = (tomo_gt_skel * tomo_d).sum() / tomo_gt_skel.sum()

    return 2*(tp*ts) / (tp + ts), tp, ts, tomo_skel, tomo_gt_skel


