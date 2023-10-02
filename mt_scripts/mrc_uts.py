import mrcfile
import numpy as np


def load_mrc(fname, mmap=False, swapxz=True):
    """
    Load an input MRC tomogram as ndarray
    :param fname: the input MRC
    :param mmap: if True (default) load a numpy memory map instead of an ndarray
    :param swapxz: if True then X and Z axis are swapped for compatibility with IMOD tomograms.
                   Don't use it with mmaps
    :return: a ndarray or a memmap if mmap is True
    """
    if mmap:
        mrc = mrcfile.mmap(fname, permissive=True)
    else:
        mrc = mrcfile.open(fname, permissive=True)
    if swapxz:
        return np.swapaxes(mrc.data, 0, 2)
    else:
        return np.swapaxes(mrc.data, 0, 2)


def write_mrc(tomo, fname, v_size=1, dtype=None, swapxz=True):
    """
    Saves a tomo (3D dataset) as MRC file
    :param tomo: tomo to save as ndarray
    :param fname: output file path
    :param v_size: voxel size (default 1)
    :param dtype: data type (default None, then the dtype of tomo is considered)
    :param swapxz: if True then X and Z axis are swapped for compatibility with IMOD tomograms.
                   Don't use it with mmaps
    :return: a numpy memory map is mmap is True, otherwise None
    """
    with mrcfile.new(fname, overwrite=True) as mrc:
        if swapxz:
            tomo = np.swapaxes(tomo, 0, 2)
        if dtype is None:
            mrc.set_data(tomo)
        else:
            mrc.set_data(tomo.astype(dtype))
        mrc.voxel_size.flags.writeable = True
        mrc.voxel_size = (v_size, v_size, v_size)
        mrc.set_volume()
        # mrc.header.ispg = 401