"""
Script for converting an SHREC text file with a particle list into and DeepFinder XML objectlist
"""

import os
import deepfinder.utils.objl as ol

import mrcfile
import numpy as np

# Functions


def read_particle_txt(filename, tomo_idx=None, lbls_list=None, cluster_size=None, bg_list=None):
    """
    Read a text file with a particle list
    :param filename: path to a text filename
    :param tomo_id: if not None (default) the 'tomo_idx' is filled with this value
    :param lbls_list: allows to fill 'class_label' integer label from string labels, if given (defatult None)
                      then particles with labels not included in the list are discarded
    :param cluster_size: if not None (default) the 'cluster_size' is filled with this value
    :param bg_list: if not None (default) add the classes list to the list as background particles
    :return: a deep-finder objetlist
    """
    objl_out, obj_id = [], 0
    with open(str(filename), 'rU') as f:
        for line in f:
            lbl, x, y, z, phi, psi, the = line.rstrip('\n').split()
            if lbls_list is not None:
                try:
                    lbl = 1 + lbls_list.index(lbl)
                except ValueError:
                    if (bg_list is not None) and (lbl in bg_list):
                        lbl = 0
                    else:
                        lbl = None
            if lbl is not None:
                hold_obj = {'tomo_idx': tomo_idx,
                            'obj_id': obj_id,
                            'label': lbl,
                            'x': float(x),
                            'y': float(y),
                            'z': float(z),
                            'psi': float(phi),
                            'phi': float(psi),
                            'the': float(the),
                            'cluster_size': cluster_size}
                objl_out.append(hold_obj)
                obj_id += 1
    return objl_out


# Main program

t_idx = 9
in_txt = '/media/martinez/Sistema/Users/Antonio/workspace/pycharm_proj/shrec_data/shrec21_full_dataset/model_' \
         + str(t_idx) + '/particle_locations.txt'
out_xml = '/media/martinez/Sistema/Users/Antonio/workspace/pycharm_proj/shrec_data/shrec21_full_dataset/model_' \
          + str(t_idx) + '/particle_locations.xml'
labels_list = ['4CR2', ] # ['4CR2', '1QVR', '1BXN', '3CF3', '1U6G', '3D2F', '2CG9', '3H84', '3GL1', '3QM1', '1S3X', '5MRC']
bg_list = ['5MRC', ]
fg_shrec_lbls_dic = {1:1, } # {1:1, 2:2, 3:3, 4:4, 5:5, 6:6, 7:7, 8:8, 9:9, 10:10, 11:11, 12:12}
in_ctomo = '/media/martinez/Sistema/Users/Antonio/workspace/pycharm_proj/shrec_data/shrec21_full_dataset/model_' \
           + str(t_idx) + '/class_mask.mrc'

objl = read_particle_txt(in_txt, tomo_idx=t_idx, lbls_list=labels_list, bg_list=bg_list)
ol.write_xml(objl, out_xml)

if fg_shrec_lbls_dic is not None:
    # Loading the SHREC class_mask.mrc
    ctomo = mrcfile.read(in_ctomo)
    # Setting FG
    fg_tomo = np.zeros(shape=ctomo.shape, dtype=np.float32)
    for shrec_lbl, fg_lbl in zip(fg_shrec_lbls_dic.keys(), fg_shrec_lbls_dic.values()):
        fg_tomo[ctomo == shrec_lbl] = fg_lbl
    # Saving FG tomogram
    mrcfile.write(os.path.split(in_ctomo)[0] + '/class_mask_fg_4CR2.mrc', fg_tomo, overwrite=True)



