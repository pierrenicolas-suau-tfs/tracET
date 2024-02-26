import scipy
import numpy as np
from src.tracET.core import diff3d, nonmaxsup_surf, nonmaxsup_line, nonmaxsup_point, angauss
from supression import desyevv


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

    return nonmaxsup_surf(tomo_l, mask,tomo_v1x, tomo_v1y, tomo_v1z)



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

def prepare_input(tomo, sigma=3,bin=False, imf=None):
    """
    Function to adapt the output for analisis. If is a binary map segmentation, it engross as a distance transformation.
    If not, it could apply a mask.
    :param tomo: Input: A scalar or binary map
    :param sigma: Standar desviation for the gaussian filter
    :param bin: True if the input is a binary map
    :param imf: Threshold for filter masc
    :return: A scalar map after mask and gaussian filters.
    """
    if bin:
        tomo_seg = tomo > 0
        tomo_dsts = scipy.ndimage.morphology.distance_transform_edt(tomo_seg)
        tomo_dsts = angauss(tomo_dsts,sigma)
        mask = np.zeros_like(tomo)
        if imf is None:
            mask[tomo_dsts > 0] = 1
        else:
            mask[tomo_dsts > imf] = 1
        tomo_dsts = tomo_dsts * mask
    else:
        if imf == None:
            mask = np.ones_like(tomo)
        else:
            mask = np.zeros_like(tomo)
            mask[tomo > imf] = 1
        tomo_dsts = angauss(tomo * mask,sigma)
    return (tomo_dsts)

def cs_dice(tomo: np.ndarray, tomo_gt: np.ndarray,sigma=3,tomo_bin=False,tomo_imf=None,gt_bin=False,gt_imf=None, dilation=0) -> tuple:
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
    tomo_dsts=prepare_input(tomo,sigma,tomo_bin,tomo_imf).astype(np.float32)

    tomo_skel, tomo_l1, tomo_v1x, tomo_v1y, tomo_v1z = surface_skel(tomo_dsts, tomo_dsts > 0)

    del tomo_dsts
    tomo_gt_dsts = prepare_input(tomo, sigma, gt_bin, gt_imf).astype(np.float32)

    tomo_gt_skel, tomo_l1, tomo_v1x, tomo_v1y, tomo_v1z = surface_skel(tomo_gt_dsts, tomo_gt_dsts > 0)

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

def cl_dice(tomo: np.ndarray, tomo_gt: np.ndarray,sigma=3,tomo_bin=False,tomo_imf=None,gt_bin=False,gt_imf=None, dilation=0) -> tuple:
    """
    Computes centerline DICE metric (cl-DICE) for two input segmented tomograms
    :param tomo: input predicted tomogoram (values >0 are considered foreground)
    :param tomo_gt: input ground truth (values >0 are considered foreground)
    :param sigma:
    :param tomo_bin:
    :param tomo_imf:
    :param gt_bin:
    :param gt_imf:
    :param dilation: number of iterations to dilate the segmentation (default 0)
    :return: returns a 3-tuple where the 1st value is cl-DICE, 2nd TP (Topology Precision), and TS
             (Topology Sensitivity)
    """
    assert tomo.shape == tomo_gt.shape



    # Getting segmentations ridges
    tomo_dsts=prepare_input(tomo,sigma,tomo_bin,tomo_imf).astype(np.float32)
    tomo_gt_dsts = prepare_input(tomo, sigma, gt_bin, gt_imf).astype(np.float32)

    tomo_skel, tomo_l1, tomo_v1x, tomo_v1y, tomo_v1z, tomo_v2x, tomo_v2y, tomo_v2z = line_skel(tomo_dsts, tomo_dsts > 0)

    del tomo_dsts
    tomo_gt_skel, tomo_l1, tomo_v1x, tomo_v1y, tomo_v1z, tomo_v2x, tomo_v2y, tomo_v2z = line_skel(tomo_gt_dsts, tomo_gt_dsts > 0)

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

def pt_dice(tomo: np.ndarray, tomo_gt: np.ndarray,sigma=3,tomo_bin=False,tomo_imf=None,gt_bin=False,gt_imf=None, dilation=0) -> tuple:
    """
    Computes point DICE metric (pt-DICE) for two input segmented tomograms
    :param tomo: input predicted tomogoram (values >0 are considered foreground)
    :param tomo_gt: input ground truth (values >0 are considered foreground)
    :param dilation: number of iterations to dilate the segmentation (default 0)
    :return: returns a 3-tuple where the 1st value is pt-DICE, 2nd TP (Topology Precision), and TS
             (Topology Sensitivity)
    """
    assert tomo.shape == tomo_gt.shape


    # Getting segmentations ridges
    tomo_dsts=prepare_input(tomo,sigma,tomo_bin,tomo_imf).astype(np.float32)
    tomo_gt_dsts = prepare_input(tomo, sigma, gt_bin, gt_imf).astype(np.float32)
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