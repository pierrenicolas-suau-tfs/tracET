"""
Script for converting al tomograms with MTs in CSV file into a scala map representation, every pixel value contains
the value linearly related with its distance to the closes MT centerline point
"""

import os
import random

import math
import numpy
import numpy as np
import pandas as pd
import scipy as sp

from mt.representation import points_to_btomo, seg_dist_trans, gauss_decay
from mt.lio import load_mrc, write_mrc
import deepfinder.utils.objl as ol


ROOT_PATH = '/media/martinez/Sistema/Users/Antonio/workspace/data/mt_nih'
in_csv = ROOT_PATH + '/tubules_norm/normal_Interpolation_10_voxels.csv'
tomo_dir = ROOT_PATH + '/tubules_norm'
out_dir = ROOT_PATH + '/out_sfields_10_voxels'
sigma = 8  # voxels
order = 2 # Gaussian derivative order
mt_dst = 12 # 14  # voxels
add_bg = True # If activated generates a background labeled object for each MT point (if possible)
binning = None # If None it is deactivated
train_set = {'0_25h_20220420_333c_tomo09',
             '1h_20211222_233a_tomo08',
             '3h_20220324_284b_tomo01',
              '3h_20220324_284b_tomo03',
             'Ctrl_20220511_368d_tomo01'}
valid_set = {'0_25h_20220420_333c_tomo04',
             'Ctrl_20220511_368d_tomo06'} # {'0_25h_20220420_333c_tomo09', '0_25h_20220420_333c_tomo04'}

# TM Background
cc_dir = ROOT_PATH + '/tm'
peaks_min_dst = 56 # voxels


# Functions
def points_to_deepfinder_objectlist(obj_list, coordinates, tomo_id: int, class_lbl: int, min_dst: int=0) -> list:
    """
    Generates the DeepFinder compatible xml from a list of coordinates
    :param obj_list: input list where the objects will be added, if None then a new empty is created
    :param coordinates: intput list of coordinates
    :param tomo_id: tomogram ID
    :param class_lbl: label class associated
    :return: a DeepFinder objectlist (a list or rows as dictionaries)
    """
    tomo_id = int(tomo_id)
    class_lbl = int(class_lbl)
    if obj_list is None:
        objl_out = list()
        obj_id = 0
    else:
        objl_out = obj_list
        obj_id = find_max_object_id(objl_out)

    for point in coordinates:
        hold_obj = {'tomo_idx': tomo_id,
                    'obj_id': obj_id,
                    'label': class_lbl,
                    'x': float(point[0]),
                    'y': float(point[1]),
                    'z': float(point[2]),
                    'psi': None,
                    'phi': None,
                    'the': None,
                    'cluster_size': None}
        objl_out.append(hold_obj)
        obj_id += 1
    return objl_out


def find_max_object_id(obj_list: list) -> int:
    """
    Find the maximum object ID of an object list
    :param obj_list: input object list
    :return: integer with the maximum obj_id
    """
    hold_max = 0
    for row in obj_list:
        if row['obj_id'] > hold_max:
            hold_max = row['obj_id']
    return hold_max


def add_bg_objects(obj_list: list, tomo: numpy.ndarray, tomo_id: int, n_points: int,
                   tomo_cc: numpy.ndarray = None, min_dist: float = 1.):
    """
    Add background labeled objects to an input object list.
    :param obj_list: input object list
    :param tomo: binary segmented tomogram with background voxel as False
    :param tomo_id: tomogram ID
    :param n_points: number of points to add, that will be set to number of background voxel if it would be set with
                     a larger value
    :param tomo_cc: cross-correlation tomogram priorize background objects selection (default None)
    :param min_dist: minium distance in voxels among selected background particles (default 1.)
    :return: None, the new objects are added to the input list
    """
    assert (len(tomo.shape) == 3) and (tomo.dtype == bool)
    if tomo_cc is not None:
        assert (min_dist is not None) and (min_dist >= 0)
        # Cross-correlation biased background selection
        tomo_cc_bg = np.invert(tomo).astype(tomo_cc.dtype) * tomo_cc
        bg_ids = find_n_peaks(tomo_cc_bg, n_points, min_dist)
    else:
        # Random background selection
        bg_ids = np.where(np.invert(tomo))
    total_bg_points = len(bg_ids[0])
    if n_points > total_bg_points:
        n_points = total_bg_points

    # Loop for points generation
    obj_id, added_ids = find_max_object_id(obj_list), list()
    for i in range(n_points):
        rand_id = random.randint(0, total_bg_points-1)
        if not(rand_id in added_ids):
            hold_obj = {'tomo_idx': tomo_id,
                        'obj_id': obj_id,
                        'label': 0,
                        'x': float(bg_ids[0][rand_id]),
                        'y': float(bg_ids[1][rand_id]),
                        'z': float(bg_ids[2][rand_id]),
                        'psi': None,
                        'phi': None,
                        'the': None,
                        'cluster_size': None}
            obj_list.append(hold_obj)
            added_ids.append(rand_id)
            obj_id += 1


def find_n_peaks(tomo, n_peaks, min_dst, v_size=1):
    """
    Find the 'n_peaks' highest peaks
    :param tomo: input tomo
    :param n_peaks: number of peaks to get
    :param min_dst: minimum distance between two peaks
    :param v_size: voxel size (default 1)
    :return: a 3-tuple witn arrays for the peaks coordinates , its length is 'n_peaks' as maximum
             Note: with respect pytmatch function, this output has been modified to replicate numpy.where() format
    """
    assert isinstance(tomo, np.ndarray) and len(tomo.shape) == 3
    assert (n_peaks >= 0) and (min_dst >= 0)

    # Tomogram sorting by their voxel values
    tomo_shape = np.asarray(tomo.shape, dtype=int)
    max_tomo_dst = math.sqrt((tomo_shape * tomo_shape).sum()) * v_size
    tomo_sort = np.argsort(tomo.flatten())[::-1]

    # Getting the highest 'n_peaks' separated by 'min_dst'
    i, l_peaks_x, l_peaks_y, l_peaks_z = 0, list(), list(), list()
    n_voxels = len(tomo_sort)
    while (i < n_voxels) and (n_peaks > len(l_peaks_x)):
        idx = tomo_sort[i]
        new_peak = np.asarray(np.unravel_index(idx, tomo_shape), dtype=float)
        # Check if the new peak is separated at least by 'min_dst' from the already added peaks
        if len(l_peaks_x) > 0:
            hold_arr_x, hold_arr_y, hold_arr_z = np.asarray(l_peaks_x), np.asarray(l_peaks_y), np.asarray(l_peaks_z)
            dsts = np.sqrt((hold_arr_x - new_peak[0])**2 + (hold_arr_y - new_peak[1])**2 + (hold_arr_z - new_peak[2])**2).sum()
            if np.min(dsts) > min_dst:
                l_peaks_x.append(new_peak[0])
                l_peaks_y.append(new_peak[1])
                l_peaks_z.append(new_peak[2])
        else:
            l_peaks_x.append(new_peak[0])
            l_peaks_y.append(new_peak[1])
            l_peaks_z.append(new_peak[2])
        i += 1

    # Converting the output to numpy.where format
    l_peaks = (np.asarray(l_peaks_x), np.asarray(l_peaks_y), np.asarray(l_peaks_z))

    return l_peaks


# Main Process

# Loop for processing tomograms
zf = 1cd
if binning is not None:
    zf = 1 / binning
df = pd.read_csv(in_csv)
tomos_set = sorted(set(df['Tomogram'].tolist()))
idx = 0
objects_list_train, objects_list_valid = None, None
for tomo_name in tomos_set:

    tomo_bare_name = tomo_name[:-7]
    tomo_fname = tomo_dir + '/' + tomo_bare_name + '.mrc'
    try:
        tomo_in = load_mrc(tomo_fname)
    except FileNotFoundError:
        print('WARNING: The file in path:', tomo_fname, 'does not exist')
        print('Continuing...')
        continue
    if binning is not None:
        tomo_in = sp.ndimage.zoom(tomo_in, zf, grid_mode=False)
    print('Processing tomogram:', tomo_name)
    df_tomo = df[df['Tomogram'] == tomo_name]

    # Loop for all point centerlines for all MTs in a tomogram
    points = list()
    for row in df_tomo.iterrows():
        points.append((row[1]['XCoord'] * zf, row[1]['YCoord'] * zf, row[1]['ZCoord'] * zf))
    if tomo_bare_name in valid_set:
        objects_list_valid = points_to_deepfinder_objectlist(objects_list_valid, points, idx, 1)
    elif tomo_bare_name in train_set:
        objects_list_train = points_to_deepfinder_objectlist(objects_list_train, points, idx, 1)
    else:
        print('\tTomogram:', tomo_fname, ' neither training nor valid set!')
        print('\tContinuing...')
        continue

    # Generating the distance map
    tomo_lbl = np.zeros(shape=tomo_in.shape, dtype=bool)
    points_to_btomo(points, tomo_lbl, True)
    tomo_dst = seg_dist_trans(tomo_lbl)
    tomo_mt = tomo_dst <= mt_dst * zf
    tomo_mt_dst = tomo_dst - mt_dst
    tomo_mt_dst[tomo_mt_dst > 0] = 0
    tomo_mt_dst *= -1

    # Background points
    if add_bg:
        if cc_dir is not None:
            tomo_cc = load_mrc(cc_dir + '/' + tomo_bare_name + '_cc.mrc')
        else:
            tomo_cc = None
        if tomo_bare_name in valid_set:
            add_bg_objects(objects_list_valid, tomo_lbl, idx, len(points), tomo_cc=tomo_cc, min_dist=peaks_min_dst)
        else:
            add_bg_objects(objects_list_train, tomo_lbl, idx, len(points), tomo_cc=tomo_cc, min_dist=peaks_min_dst)

    # Gaussian decay
    tomo_decay = gauss_decay(tomo_dst, sigma, order)
    if order == 2:
        # Invert and set negative values to zero
        tomo_decay *= -1
        tomo_decay[tomo_decay < 0] = .0

    # Store the output
    out_tomo = os.path.splitext(tomo_name)[0]
    # write_mrc(tomo_dst.astype(np.float32), out_dir + '/' + out_tomo + '_idx_' + str(idx) + '_dst.mrc')
    write_mrc((tomo_mt).astype(np.float32), out_dir + '/' + out_tomo + '_idx_' + str(idx) + '_mt.mrc')
    write_mrc((tomo_mt_dst).astype(np.float32), out_dir + '/' + out_tomo + '_idx_' + str(idx) + '_mt_dst.mrc')
    # write_mrc(tomo_decay.astype(np.float32), out_dir + '/' + out_tomo + '_idx_' + str(idx) + '_decay.mrc')
    if binning:
        write_mrc(tomo_in.astype(np.float32), out_dir + '/' + out_tomo + '_idx_' + str(idx) + '_bin' + str(binning) + '.mrc')
    # else:
    #     write_mrc(tomo_in.astype(np.float32), out_dir + '/' + out_tomo + '_idx_' + str(idx) + '.mrc')
    idx += 1

if add_bg:
    ol.write_xml(objects_list_train, out_dir + '/train_objlist_bg_tm.xml')
    ol.write_xml(objects_list_valid, out_dir + '/valid_objlist_bg_tm.xml')
else:
    ol.write_xml(objects_list_train, out_dir + '/train_objlist.xml')
    ol.write_xml(objects_list_valid, out_dir + '/valid_objlist.xml')