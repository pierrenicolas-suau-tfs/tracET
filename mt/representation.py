"""
Module with functionality to represent structures in cryo-ET with different formats and covert data between them
"""

__author__ = 'Antonio Martinez-Sanchez (anmartinezs@um.es)'

import math

# Imports
import scipy
import numpy as np

# Module variable
MOD_DEBUG = True

# Functions


def points_to_btomo(points, tomo, lbl):
    """
    Maps a set of points (3D coordinates) in a tomogram
    :param points: iterable with points (3D coordinates), non integer coordinates are rounded
    :param tomo: binary map (3D ndarray) to map the points with lbl
    :param lbl: output label for the points
    :return: the tomogram with the points labelled. IndexError are handled so points out of input tomo bounds are
             not labelled.
    """
    assert hasattr(points, '__len__')
    if len(points) > 0:
        assert hasattr(points[0], '__len__') and (len(points[0]) == 3)

    for point in points:
        x, y, z = int(round(point[0])), int(round(point[1])), int(round(point[2]))
        try:
            tomo[x, y, z] = lbl
        except IndexError:
            if MOD_DEBUG:
                print('WARNING: points_to_debug: Point out of tomogram bounds!')
            continue

    return tomo


def seg_dist_trans(seg):
    """
    Distance transform to a segmented (binary) tomogram
    :param seg: input segmented tomogram, if not binary then every pixel greater than 0 is considered as segmented
                foreground.
    :return: a tomogram with the same size as input where pixels correspond with the eculidean pixel distance to
             the closest segmented tomogram
    """

    if seg.dtype != bool:
        tomod = scipy.ndimage.morphology.distance_transform_edt(np.invert(seg > 0))
    else:
        tomod = scipy.ndimage.morphology.distance_transform_edt(np.invert(seg))

    return tomod


def exp_decay(sfield, tau=1):
    """
    Applies and exponential decay to a scalar field N(s)=e^(-s/tau)
    :param sfield: tomogram (3D ndarray) with the input scalar field
    :param tau: scaling term, that is the value where the input values are scaled by factor 1/e
    :return: output tomogram
    """
    assert tau > 0
    assert isinstance(sfield, np.ndarray) and (len(sfield.shape) == 3)
    return np.exp(-sfield/tau)


def gauss_decay(sfield, sigma=1, d_order=0):
    """
    Applies and Gaussian decay to a scalar field N(s)=e^(-s²/(2*s^2))
    :param sfield: tomogram (3D ndarray) with the input scalar field
    :param tau: scaling term, that is the value where the input values are scaled by factor 1/e
    :param d_order: derivate order (default 0), valid on up to 2.
    :return: output tomogram
    """
    assert sigma > 0
    assert isinstance(sfield, np.ndarray) and (len(sfield.shape) == 3)
    assert (d_order >= 0) and (d_order <= 2)
    sigma_2 = sigma * sigma
    gauss_0 = (1/(sigma*math.sqrt(2*math.pi))) * np.exp(-(sfield*sfield) / (2*sigma_2))
    if d_order == 1:
        return -(sfield/sigma_2) * gauss_0
    elif d_order == 2:
        return ((sfield*sfield - sigma_2) / (sigma_2 * sigma_2)) * gauss_0
    else:
        return gauss_0

