
import numpy as np
from tracET.core.diff import diff3d, nonmaxsup_surf, nonmaxsup_line, nonmaxsup_point
from supression import desyevv


def surface_skel(tomo: np.ndarray,f=0) -> np.ndarray:
    """
    From an input tomogram compute its skeleton for surface ridges
    :param tomo: input tomogram
    :param mask: Default None, if given binary mask (np.ndarray) to only consider ridges within the 1-valued voxles
    :return: a binary tomogram with the skeleton
    """
    #if mask is None:
        #mask = np.ones(shape=tomo.shape, dtype=bool)
    #else:
        #assert mask.shape == tomo.shape
    [Nx, Ny, Nz] = np.shape(tomo)
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
    tomo_l = np.swapaxes(np.reshape(-tomo_l, (Nz, Ny, Nx)), 0, 2)
    tomo_v1x = np.swapaxes(np.reshape(tomo_v1x, (Nz, Ny, Nx)), 0, 2)
    tomo_v1y = np.swapaxes(np.reshape(tomo_v1y, (Nz, Ny, Nx)), 0, 2)
    tomo_v1z = np.swapaxes(np.reshape(tomo_v1z, (Nz, Ny, Nx)), 0, 2)
    # Non-maximum suppression

    return nonmaxsup_surf(tomo_l, tomo_l>f,tomo_v1x, tomo_v1y, tomo_v1z)



#def line_skel(tomo: np.ndarray, mask=None, mode='hessian') -> np.ndarray:
def line_skel(tomo: np.ndarray, f=1, mode='hessian') -> np.ndarray:
    """
    From an input tomogram compute its skeleton for line ridges
    :param tomo: input tomogram
    :param mask: Default None, if given binary mask (np.ndarray) to only consider ridges within the 1-valued voxles
    :param mode: for computing the eigenvalues the Hessian tensor is always used, but for eigenvectors if 'hessian'
                 (default) then the Hessian tensor if 'structure' then the Structure tensor is used
    :return: a binary tomogram with the skeleton
    """
    #if mask is None:
    #    mask = np.ones(shape=tomo.shape, dtype=bool)
    #else:
    #    assert mask.shape == tomo.shape
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
        tomo_l1 = np.swapaxes(np.reshape(-tomo_l1, (Nz, Ny, Nx)), 0, 2)
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
    return nonmaxsup_line(tomo_l1, tomo_l1>f, tomo_v1x, tomo_v1y, tomo_v1z, tomo_v2x, tomo_v2y, tomo_v2z)


#def point_skel(tomo: np.ndarray, mask=None, mode='hessian') -> np.ndarray:
def point_skel(tomo: np.ndarray, f=1, mode='hessian') -> np.ndarray:
    """
    From an input tomogram compute its skeleton for pint ridges
    :param tomo: input tomogram
    :param mask: Default None, if given binary mask (np.ndarray) to only consider ridges within the 1-valued voxles
    :param mode: for computing the eigenvalues the Hessian tensor is always used, but for eigenvectors if 'hessian'
                 (default) then the Hessian tensor if 'structure' then the Structure tensor is used
    :return: a binary tomogram with the skeleton
    """
   # if mask is None:
   #     mask = np.ones(shape=tomo.shape, dtype=bool)
   # else:
   #     assert mask.shape == tomo.shape
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
    (tomo_l1, _, _,
     tomo_v1x, tomo_v1y, tomo_v1z,
     tomo_v2x, tomo_v2y, tomo_v2z,
     tomo_v3x, tomo_v3y, tomo_v3z) = desyevv(tomo_xx, tomo_yy, tomo_zz, tomo_xy, tomo_xz, tomo_yz)
    if mode != 'structure':

        tomo_l1 = np.swapaxes(np.reshape(-tomo_l1, (Nz, Ny, Nx)), 0, 2)
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

        tomo_l1 = np.swapaxes(np.reshape(-tomo_l1, (Nz, Ny, Nx)), 0, 2)
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
    return nonmaxsup_point(tomo_l1,tomo_l1>f, tomo_v1x, tomo_v1y, tomo_v1z, tomo_v2x, tomo_v2y, tomo_v2z, tomo_v3x, tomo_v3y, tomo_v3z)