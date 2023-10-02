"""
Find local maxima (higher value than neighborhood) in a tomogram
"""

__author__ = 'Antonio Martínez-Sánchez'

import itertools
import nrrd
import numpy as np
from scipy import ndimage
import sys, getopt, os, time, csv

from mt import lio
from mt import utils


def find_local_maxima(tomo, conn=26):
    """
    Find all local minima, a voxel with a higher value than its neighbors, in a tomogram
    :param tomo: input tomo
    :param conn: neighborhood connectivity, valid: 6, 18, 26 (default)
    :return: a binary tomogram with local minima 1-valued
    """
    assert isinstance(tomo, np.ndarray) and len(tomo.shape) == 3
    assert (conn == 6) or (conn == 18) or (conn == 26)
    cont, total = 0, np.prod(np.asarray(tomo.shape)-2)

    tomo_lmb = np.zeros(shape=tomo.shape, dtype=bool)
    if conn == 6:
        for i, j, k in itertools.product(range(1,tomo.shape[0]-1), range(1,tomo.shape[1]-1), range(1,tomo.shape[2]-1)):
            v = tomo[i, j, k]
            v_1, v_2, v_3 = tomo[i - 1, j, k], tomo[i, j - 1, k], tomo[i, j, k - 1]
            v_4, v_5, v_6 = tomo[i + 1, j, k], tomo[i, j + 1, k], tomo[i, j, k + 1]
            if (v > v_1) and (v > v_2) and (v > v_3) and (v > v_4) and (v > v_5) and (v > v_6):
                tomo_lmb[i, j, k] = True
    if conn == 18:
        for i, j, k in itertools.product(range(1,tomo.shape[0]-1), range(1,tomo.shape[1]-1), range(1,tomo.shape[2]-1)):
            v = tomo[i, j, k]
            v_1, v_2, v_3 = tomo[i - 1, j, k], tomo[i, j - 1, k], tomo[i, j, k - 1]
            v_4, v_5, v_6 = tomo[i + 1, j, k], tomo[i, j + 1, k], tomo[i, j, k + 1]
            v_7, v_8, v_9, v_10 = tomo[i - 1, j - 1, k], tomo[i - 1, j + 1, k], tomo[i + 1, j - 1, k], tomo[i + 1, j + 1, k]
            v_11, v_12, v_13, v_14 = tomo[i - 1, j, k - 1], tomo[i - 1, j, k + 1], tomo[i + 1, j, k - 1], tomo[i + 1, j, k + 1]
            v_15, v_16, v_17, v_18 = tomo[i, j - 1, k - 1], tomo[i, j - 1, k + 1], tomo[i, j + 1, k - 1], tomo[i, j + 1, k + 1]
            if (v > v_1) and (v > v_2) and (v > v_3) and (v > v_4) and (v > v_5) and (v > v_6) and (v > v_7) and \
                (v > v_8) and (v > v_9) and (v > v_10) and (v > v_11) and (v > v_12) and (v > v_13) and (v > v_14) and \
                (v > v_15) and (v > v_16) and (v > v_17) and (v > v_18):
                    tomo_lmb[i, j, k] = True
    else:
        for i, j, k in itertools.product(range(1,tomo.shape[0]-1), range(1,tomo.shape[1]-1), range(1,tomo.shape[2]-1)):
            v = tomo[i, j, k]
            v_1, v_2, v_3 = tomo[i - 1, j, k], tomo[i, j - 1, k], tomo[i, j, k - 1]
            v_4, v_5, v_6 = tomo[i + 1, j, k], tomo[i, j + 1, k], tomo[i, j, k + 1]
            v_7, v_8, v_9, v_10 = tomo[i - 1, j - 1, k], tomo[i - 1, j + 1, k], tomo[i + 1, j - 1, k], tomo[i + 1, j + 1, k]
            v_11, v_12, v_13, v_14 = tomo[i - 1, j, k - 1], tomo[i - 1, j, k + 1], tomo[i + 1, j, k - 1], tomo[i + 1, j, k + 1]
            v_15, v_16, v_17, v_18 = tomo[i, j - 1, k - 1], tomo[i, j - 1, k + 1], tomo[i, j + 1, k - 1], tomo[i, j + 1, k + 1]
            v_19, v_20, v_21, v_22 = tomo[i - 1, j - 1, k - 1], tomo[i - 1, j - 1, k + 1], tomo[i - 1, j + 1, k - 1], tomo[i - 1, j + 1, k + 1]
            v_23, v_24, v_25, v_26 = tomo[i + 1, j - 1, k - 1], tomo[i + 1, j - 1, k + 1], tomo[i + 1, j + 1, k - 1], tomo[i + 1, j + 1, k + 1]
            if (v > v_1) and (v > v_2) and (v > v_3) and (v > v_4) and (v > v_5) and (v > v_6) and (v > v_7) and \
                (v > v_8) and (v > v_9) and (v > v_10) and (v > v_11) and (v > v_12) and (v > v_13) and (v > v_14) and \
                (v > v_15) and (v > v_16) and (v > v_17) and (v > v_18) and (v > v_19) and (v > v_20) and \
                (v > v_21) and (v > v_22) and (v > v_23) and (v > v_24) and (v > v_25) and (v > v_26):
                    tomo_lmb[i, j, k] = True
            cont += 1
            print('\t\t+Processing', cont, 'of', total)

    return tomo_lmb


def bin_tomo_list(tomo_bin):
    """
    Convert a binary tomogram into a list of coordinates
    :param tomo_bin: input binary tomogram
    :return: a list with the coordinates of the True valued voxles
    """
    assert isinstance(tomo_bin, np.ndarray) and len(tomo_bin.shape) == 3 and tomo_bin.dtype == bool

    out_list = list()
    hold_list = np.where(tomo_bin)
    for i in range(len(hold_list[0])):
        out_list.append(np.asarray((hold_list[0][i], hold_list[1][i], hold_list[2][i]), dtype=float))

    return out_list


def save_list_coords(coords, fname):
    """
    Save a list of coordinates in a CSV file
    :param coords: list of coordinates
    :param fname: output filename
    :return:
    """
    with open(fname, 'w', newline='') as csvfile:
        fieldnames = ['x', 'y', 'z']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for coord in coords:
            writer.writerow({'x':coord[0], 'y':coord[1], 'z':coord[2]})


def gen_smap_list(coords, smap):
    """
    Generates a list with values taken from a scalar map on a given list coordinates
    :param coords: input list of coordinates to sort
    :param smap: scalar map
    :return: an array with the scalar map values for the input given coordinates
    """
    # Sampling the scalar map values
    corrs = np.zeros(shape=len(coords), dtype=float)
    for i, coord in enumerate(coords):
        corrs[i] = utils.trilin_interp(coord[0], coord[1], coord[2], smap)
    return corrs


def main(argv):

    # Input parsing
    in_tomo, out_tomo, out_csv, s_log = None, None, None, None
    dist = None
    try:
        opts, args = getopt.getopt(argv, 'hi:o:c:s:d:', ['help', 'itomo', 'otomo', 'ocsv', 'slog', 'dist'])
    except getopt.GetoptError:
        print('python lm_tomo.py -i <in_tomo> -o <out_tomo>')
        sys.exit()
    for opt, arg in opts:
        if opt == '-h':
            print('python lm_tomo.py -i <particle_list> -o <out_tomo>')
            print('\t-i (--itomo) <in_tomo>: input tomogram (scalar map)')
            print('\t-o (--otomo) <out_tomo>: output binary tomgram (1-local maxima)')
            print('\t-c (--ocsv) <out_csv> (optional): output list of local maxima as CSV file with their coordinates')
            print('\t-s (--slog) <sigma_LoG> (optional): sigma for LoG operator for blobs pre-detection.')
            print('\t-d (--dist) <min_dist> (optional): minimum distance among local maxima, only applied if option',
                  '\'c\' is also active')
            sys.exit()
        elif opt in ("-i", "--itomo"):
            in_tomo = arg
            if not(os.path.splitext(in_tomo)[1] in ('.mrc', '.nhdr', '.nrrd')):
                print('The input file must have a .mrc, .nhdr or nrrd extension!')
                sys.exit()
        elif opt in ("-o", "--itomo"):
            out_tomo = arg
            if not (os.path.splitext(out_tomo)[1] in ('.mrc', '.nhdr', '.nrrd')):
                print('The input file must have a .mrc, .nhdr or nrrd extension!')
                sys.exit()
        elif opt in ("-c", "--ocsv"):
            out_csv = arg
            if not(os.path.splitext(out_csv)[1] in ('.csv')):
                print('The output file must have a .csv extension!')
                sys.exit()
        elif opt in ("-s", "--slog"):
            s_log = float(arg)
        elif opt in ("-d", "--dist"):
            dist = float(arg)
    if (in_tomo is None) or (out_tomo is None):
        print('python lm_tomo.py -i <particle_list> -o <out_tomo>')
        print('\t-i (--itomo) <in_tomo>: input tomogram (scalar map)')
        print('\t-o (--otomo) <out_tomo>: output binary tomgram (1-local maxima)')
        print('\t-c (--ocsv) <out_csv>(optional): output list of local maxima as CSV file with their coordinates')
        print('\t-s (--slog) <sigma_LoG>(optional): sigma for LoG operator for blobs pre-detection.')
        print('\t-d (--dist) <min_dist> (optional): minimum distance among local maxima')
        sys.exit()

    print('\t-Loading input tomogram:', in_tomo)
    if os.path.splitext(in_tomo)[1] == '.mrc':
        tomo = lio.load_mrc(in_tomo)
    else:
        tomo = nrrd.read(in_tomo)[0]

    print('\t-Finding local minima...')
    tomo_lmb = find_local_maxima(tomo)

    if s_log is not None:
        print('\t-Computing LoG...')
        mask = ndimage.gaussian_laplace(tomo, s_log) < 0
        tomo_lmb *= mask

    print('\t-Storing the output file:', out_tomo)
    if os.path.splitext(out_tomo)[1] == '.mrc':
        lio.write_mrc(np.asarray(tomo_lmb, dtype=np.int8), out_tomo)
    else:
        nrrd.write(out_tomo, np.asarray(tomo_lmb, dtype=np.int8))

    if out_csv is not None:
        print('\t-Storing the output CSV file:', out_csv)
        list_lmb = bin_tomo_list(tomo_lmb)
        if dist is not None:
            print('\t-Keeping maxima with the highest scalar values to ensure a minimum distance of', str(dist))
            corrs = gen_smap_list(list_lmb, tomo)
            list_lmb = utils.coords_scale_supression(list_lmb, dist, weights=corrs, filter=True)
        save_list_coords(list_lmb, out_csv)

    print('Successfully terminated. (' + time.strftime("%c") + ')')


if __name__ == "__main__":
    main(sys.argv[1:])
