"""
Function implementing differentioal geometry operations on tomograms
"""

import numpy as np
from supression import desyevv, nonmaxsup_0, nonmaxsup_1, nonmaxsup_2


def angauss(I, s, r=1):
    """

    :param I:
    :param s:
    :param r:
    :return:
    """
    # Initialitation
    [Nx, Ny, Nz] = np.shape(I)
    if np.remainder(Nx, 2) == 1:
        Nx2 = np.floor(0.5 * Nx)
        Vnx = np.arange(-Nx2, Nx2 + 1)
    else:
        Nx2 = 0.5 * Nx
        Vnx = -np.arange(-Nx2, Nx2)
    if np.remainder(Ny, 2) == 1:
        Ny2 = np.floor(0.5 * Ny)
        Vny = np.arange(-Ny2, Ny2 + 1)
    else:
        Ny2 = 0.5 * Ny
        Vny = -np.arange(-Ny2, Ny2)
    if np.remainder(Nz, 2) == 1:
        Nz2 = np.floor(0.5 * Nz)
        Vnz = np.arange(-Nz2, Nz2 + 1)
    else:
        Nz2 = 0.5 * Nz
        Vnz = -np.arange(-Nz2, Nz2)

    [X, Y, Z] = np.meshgrid(Vny, Vnx, Vnz)
    A = 1. / ((s * s * np.sqrt(r * (2. * np.pi) ** 3.)))
    a = 1. / (2. * s * s)
    b = a / r

    # Kernel
    K = A * np.exp(-a * (X * X + Y * Y) - b * Z * Z)

    # Convolution
    F = np.fft.fftshift(np.fft.ifftn(np.fft.fftn(I) * np.fft.fftn(K)))

    return F


def diff3d(T, k):
    """

    Args:
        T:
        k:

    Returns:

    """
    [Nx, Ny, Nz] = np.shape(T)
    Idp = np.zeros((Nx, Ny, Nz), dtype=T.dtype)
    Idn = np.zeros((Nx, Ny, Nz), dtype=T.dtype)

    if k == 0:
        Idp[0:Nx - 2, :, :] = T[1:Nx - 1, :, :]
        Idn[1:Nx - 1, :, :] = T[0:Nx - 2, :, :]
        # Pad extremes
        Idp[Nx - 1, :, :] = Idp[Nx - 2, :, :]
        Idp[0, :, :] = Idp[1, :, :]
    elif k == 1:
        Idp[:, 0:Ny - 2, :] = T[:, 1:Ny - 1, :]
        Idn[:, 1:Ny - 1, :] = T[:, 0:Ny - 2, :]
        # Pad extremes
        Idp[:, Ny - 1, :] = Idp[:, Ny - 2, :]
        Idp[:, 0, :] = Idp[:, 1, :]
    else:
        Idp[:, :, 0:Nz - 2] = T[:, :, 1:Nz - 1]
        Idn[:, :, 1:Nz - 1] = T[:, :, 0:Nz - 2]
        # Pad extremes
        Idp[:, :, Nz - 1] = Idp[:, :, Nz - 2]
        Idp[:, :, 0] = Idp[:, :, 1]

    return ((Idp - Idn) * 0.5)


def eig3dk(Ixx, Iyy, Izz, Ixy, Ixz, Iyz):
    """

    Args:
        Ixx:
        Iyy:
        Izz:
        Ixy:
        Ixz:
        Iyz:

    Returns:

    """

    # Falttern for C-processing
    [Nx, Ny, Nz] = np.shape(Ixx)
    Ixx = np.swapaxes(Ixx.astype(np.float32), 0, 2).flatten()
    Iyy = np.swapaxes(Iyy.astype(np.float32), 0, 2).flatten()
    Izz = np.swapaxes(Izz.astype(np.float32), 0, 2).flatten()
    Ixy = np.swapaxes(Ixy.astype(np.float32), 0, 2).flatten()
    Ixz = np.swapaxes(Ixz.astype(np.float32), 0, 2).flatten()
    Iyz = np.swapaxes(Iyz.astype(np.float32), 0, 2).flatten()

    # C-processing
    L1, L2, L3, V1x, V1y, V1z, V2x, V2y, V2z, V3x, V3y, V3z = desyevv(Ixx, Iyy, Izz, Ixy, Ixz, Iyz)
    del Ixx
    del Iyy
    del Izz
    del Ixy
    del Ixz
    del Iyz

    # Rccovering tomogram shape
    L1 = np.swapaxes(np.reshape(L1, (Nz, Ny, Nx)), 0, 2)
    L2 = np.swapaxes(np.reshape(L2, (Nz, Ny, Nx)), 0, 2)
    L3 = np.swapaxes(np.reshape(L3, (Nz, Ny, Nx)), 0, 2)

    V1x = np.swapaxes(np.reshape(V1x, (Nz, Ny, Nx)), 0, 2)
    V1y = np.swapaxes(np.reshape(V1y, (Nz, Ny, Nx)), 0, 2)
    V1z = np.swapaxes(np.reshape(V1z, (Nz, Ny, Nx)), 0, 2)

    V2x = np.swapaxes(np.reshape(V2x, (Nz, Ny, Nx)), 0, 2)
    V2y = np.swapaxes(np.reshape(V2y, (Nz, Ny, Nx)), 0, 2)
    V2z = np.swapaxes(np.reshape(V2z, (Nz, Ny, Nx)), 0, 2)

    V3x = np.swapaxes(np.reshape(V3x, (Nz, Ny, Nx)), 0, 2)
    V3y = np.swapaxes(np.reshape(V3y, (Nz, Ny, Nx)), 0, 2)
    V3z = np.swapaxes(np.reshape(V3z, (Nz, Ny, Nx)), 0, 2)

    return [L1, L2, L3, V1x, V1y, V1z, V2x, V2y, V2z, V3x, V3y, V3z]


def nonmaxsup_surf(I, M, V1x, V1y, V1z):
    """
    Applies non-maximum suppresion criteria for detecting local maxima in 2-manifolds (surfaces)
    :param I: input tomogram
    :param M: input mask (the criterion is applied only max 1-valued voxels, otherwise a zero is directly assigned)
    :param Vnm: the coordinate 'm' of the 'n' eigenvector
    :param return: a binary tomogram were 1-valued voxel are the ones fulfilling non-maximum suppression criteria
    """

    [Nx, Ny, Nz] = np.shape(I)
    H = np.zeros((Nx, Ny, Nz))
    H[1:Nx - 2, 1:Ny - 2, 1:Nz - 2] = 1
    M = M * H
    del H
    M = np.swapaxes(M, 0, 2).flatten().astype(int)
    Mr = np.arange(0, Nx * Ny * Nz, dtype=int)
    Mr = Mr[M == 1]

    Ir = np.swapaxes(I.astype(np.float32), 0, 2).flatten()
    V1xr = np.swapaxes(V1x.astype(np.float32), 0, 2).flatten()
    V1yr = np.swapaxes(V1y.astype(np.float32), 0, 2).flatten()
    V1zr = np.swapaxes(V1z.astype(np.float32), 0, 2).flatten()

    dim = np.array([Nx, Ny]).astype('uint32')

    Br = nonmaxsup_2(Ir, V1xr, V1yr, V1zr, Mr, dim)
    del Ir
    del V1xr
    del V1yr
    del V1zr
    del Mr

    B = np.swapaxes(np.reshape(Br, (Nz, Ny, Nx)), 0, 2)
    H = np.zeros((Nx, Ny, Nz))
    H[2:Nx - 2, 2:Ny - 2, 2:Nz - 2] = 1
    B = B * H
    del H

    return B


def nonmaxsup_line(I, M, V1x, V1y, V1z, V2x, V2y, V2z):
    """
    Applies non-maximum suppresion criteria for detecting local maxima in 1-manifolds (lines)
    :param I: input tomogram
    :param M: input mask (the criterion is applied only max 1-valued voxels, otherwise a zero is directly assigned)
    :param Vnm: the coordinate 'm' of the 'n' eigenvector
    :param return: a binary tomogram were 1-valued voxel are the ones fulfilling non-maximum suppression criteria
    """

    [Nx, Ny, Nz] = np.shape(I)
    H = np.zeros((Nx, Ny, Nz))
    H[1:Nx - 2, 1:Ny - 2, 1:Nz - 2] = 1
    M = M * H
    del H
    M = np.swapaxes(M, 0, 2).flatten().astype(int)
    Mr = np.arange(0, Nx * Ny * Nz, dtype=int)
    Mr = Mr[M == 1]

    Ir = np.swapaxes(I.astype(np.float32), 0, 2).flatten()
    V1xr = np.swapaxes(V1x.astype(np.float32), 0, 2).flatten()
    V1yr = np.swapaxes(V1y.astype(np.float32), 0, 2).flatten()
    V1zr = np.swapaxes(V1z.astype(np.float32), 0, 2).flatten()
    V2xr = np.swapaxes(V2x.astype(np.float32), 0, 2).flatten()
    V2yr = np.swapaxes(V2y.astype(np.float32), 0, 2).flatten()
    V2zr = np.swapaxes(V2z.astype(np.float32), 0, 2).flatten()

    dim = np.array([Nx, Ny]).astype('uint32')

    Br = nonmaxsup_1(Ir, V1xr, V1yr, V1zr, V2xr, V2yr, V2zr, Mr, dim)
    del Ir
    del V1xr
    del V1yr
    del V1zr
    del V2xr
    del V2yr
    del V2zr
    del Mr

    B = np.swapaxes(np.reshape(Br, (Nz, Ny, Nx)), 0, 2)
    H = np.zeros((Nx, Ny, Nz))
    H[2:Nx - 2, 2:Ny - 2, 2:Nz - 2] = 1
    B = B * H
    del H

    return B


def nonmaxsup_point(I, M, V1x, V1y, V1z, V2x, V2y, V2z, V3x, V3y, V3z):
    """
    Applies non-maximum suppresion criteria for detecting local maxima in 0-manifolds (points)
    :param I: input tomogram
    :param M: input mask (the criterion is applied only max 1-valued voxels, otherwise a zero is directly assigned)
    :param Vnm: the coordinate 'm' of the 'n' eigenvector
    :param return: a binary tomogram were 1-valued voxel are the ones fulfilling non-maximum suppression criteria
    """

    [Nx, Ny, Nz] = np.shape(I)
    H = np.zeros((Nx, Ny, Nz))
    H[1:Nx - 2, 1:Ny - 2, 1:Nz - 2] = 1
    M = M * H
    del H
    M = np.swapaxes(M, 0, 2).flatten().astype(int)
    Mr = np.arange(0, Nx * Ny * Nz, dtype=int)
    Mr = Mr[M == 1]

    Ir = np.swapaxes(I.astype(np.float32), 0, 2).flatten()
    V1xr = np.swapaxes(V1x.astype(np.float32), 0, 2).flatten()
    V1yr = np.swapaxes(V1y.astype(np.float32), 0, 2).flatten()
    V1zr = np.swapaxes(V1z.astype(np.float32), 0, 2).flatten()
    V2xr = np.swapaxes(V2x.astype(np.float32), 0, 2).flatten()
    V2yr = np.swapaxes(V2y.astype(np.float32), 0, 2).flatten()
    V2zr = np.swapaxes(V2z.astype(np.float32), 0, 2).flatten()
    V3xr = np.swapaxes(V2x.astype(np.float32), 0, 2).flatten()
    V3yr = np.swapaxes(V2y.astype(np.float32), 0, 2).flatten()
    V3zr = np.swapaxes(V2z.astype(np.float32), 0, 2).flatten()

    dim = np.array([Nx, Ny]).astype('uint32')

    Br = nonmaxsup_0(Ir, V1xr, V1yr, V1zr, V2xr, V2yr, V2zr, V3xr, V3yr, V3zr, Mr, dim)
    del Ir
    del V1xr
    del V1yr
    del V1zr
    del V2xr
    del V2yr
    del V2zr
    del V3xr
    del V3yr
    del V3zr
    del Mr

    B = np.swapaxes(np.reshape(Br, (Nz, Ny, Nx)), 0, 2)
    H = np.zeros((Nx, Ny, Nz))
    H[2:Nx - 2, 2:Ny - 2, 2:Nz - 2] = 1
    B = B * H
    del H

    return B