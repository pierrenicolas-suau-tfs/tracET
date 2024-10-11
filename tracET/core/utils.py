"""
Miscellaneous functions
"""

import math
import numpy as np


def add_cloud_gauss(tomo: np.ndarray, coords: np.ndarray, g_std: np.ndarray) -> np.ndarray:
    """
    Add a translated Gaussian at the specified coordinates
    :param tomo: input tomogram where the Gaussian are going to be added
    :param coords: list of coordinates for the Gaussians
    :param g_std: Gaussian standard deviation
    :return: a np.ndarray with the added Guassian density
    """

    # Mark Gaussian centers
    hold_tomo = np.zeros(shape=tomo.shape, dtype=np.float32)
    for coord in coords:
        x, y, z = int(round(coord[0])), int(round(coord[1])), int(round(coord[2]))
        hold_tomo[x, y, z] = 1

    # Building Guassian model
    nx, ny, nz = (tomo.shape[0] - 1) * .5, (tomo.shape[1] - 1) * .5, (tomo.shape[2] - 1) * .5
    if (nx % 1) == 0:
        arr_x = np.concatenate((np.arange(-nx, 0, 1), np.arange(0, nx + 1, 1)))
    else:
        if nx < 1:
            arr_x = np.arange(0, 1)
        else:
            nx = math.ceil(nx)
            arr_x = np.concatenate((np.arange(-nx, 0, 1), np.arange(0, nx, 1)))
    if (ny % 1) == 0:
        arr_y = np.concatenate((np.arange(-ny, 0, 1), np.arange(0, ny + 1, 1)))
    else:
        if ny < 1:
            arr_y = np.arange(0, 1)
        else:
            ny = math.ceil(ny)
            arr_y = np.concatenate((np.arange(-ny, 0, 1), np.arange(0, ny, 1)))
    if (nz % 1) == 0:
        arr_z = np.concatenate((np.arange(-nz, 0, 1), np.arange(0, nz + 1, 1)))
    else:
        if nz < 1:
            arr_z = np.arange(0, 1)
        else:
            nz = math.ceil(nz)
            arr_z = np.concatenate((np.arange(-nz, 0, 1), np.arange(0, nz, 1)))
    [X, Y, Z] = np.meshgrid(arr_x, arr_y, arr_z, indexing='ij')
    X = X.astype(np.float32, copy=False)
    Y = Y.astype(np.float32, copy=False)
    Z = Z.astype(np.float32, copy=False)
    R = np.sqrt(X * X + Y * Y + Z * Z)
    gauss = (1/(g_std*g_std*g_std*math.sqrt(2*np.pi))) * np.exp(-R / (2. * g_std * g_std))

    # Convolution
    tomo_conv = np.real(np.fft.ifftn(np.fft.fftn(hold_tomo) * np.fft.fftn(gauss)))

    # Adding the Gaussina to the input tomogram
    return tomo + tomo_conv

