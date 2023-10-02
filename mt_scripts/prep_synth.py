"""
Script for preparing synthetic tomograms with their motif list to be processed by lines_to_smap.py
"""

import os
import numpy as np
import scipy as sp
import pandas as pd
from mt.lio import load_mrc, write_mrc

ROOT_PATH = '/media/martinez/Sistema/Users/Antonio/workspace/data/mt_nih/synth_tubules'

folders = ['out_all_tomos_1-2',
           'out_all_tomos_3-4',
           'out_all_tomos_5-6',
           'out_all_tomos_7-8',
           'out_all_tomos_9-10']
vsize = 10 # A/voxel
vsize_exp = 10 # .68 # A/voxel


# Loop for processing the folders
out_df = None
for idx, folder in enumerate(folders):

    print('Processing folder', folder, '...')

    # Rescale the tomograms
    tomo = load_mrc(ROOT_PATH + '/' + folder + '/tomos/tomo_rec_0.mrc')
    if vsize == vsize_exp:
        write_mrc(tomo, ROOT_PATH + '/tomo_rec_' + str(idx * 2) + '.mrc')
    else:
        write_mrc(sp.ndimage.zoom(tomo, vsize/vsize_exp), ROOT_PATH + '/tomo_rec_' + str(idx * 2) + '.mrc')
    tomo = load_mrc(ROOT_PATH + '/' + folder + '/tomos/tomo_rec_1.mrc')
    if vsize == vsize_exp:
        write_mrc(tomo, ROOT_PATH + '/tomo_rec_' + str(idx * 2 + 1) + '.mrc')
    else:
        write_mrc(sp.ndimage.zoom(tomo, vsize / vsize_exp), ROOT_PATH + '/tomo_rec_' + str(idx * 2 + 1) + '.mrc')

    # Load the input motif list
    in_motif = ROOT_PATH + '/' + folder + '/tomos_motif_list.csv'
    df = pd.read_csv(in_motif, delimiter='\t')

    # Filter for selecting both ribosome-like structures
    mt_df1 = df[df['Label'] == 6]
    mt_df2 = df[df['Label'] == 11]
    mt_df = pd.concat([mt_df1, mt_df2])
    # mt_df = df[df['Label'] >= 24] # Uncomment this line and comment the previous ones to select a membrane bound proteins

    # Add columns to be processed by lines_to_smap.py
    mt_df.insert(0, 'ZCoord', np.asarray(mt_df['Z'].tolist()) / vsize_exp)
    mt_df.insert(0, 'YCoord', np.asarray(mt_df['Y'].tolist()) / vsize_exp)
    mt_df.insert(0, 'XCoord', np.asarray(mt_df['X'].tolist()) / vsize_exp)
    new_names = list()
    for row in mt_df.iterrows():
        hold_name = os.path.split(row[1]['Density'])[1]
        if hold_name == 'tomo_den_0.mrc':
            new_names.append('tomo_rec_' + str(idx * 2) + '.mrc')
        else:
            new_names.append('tomo_rec_' + str(idx * 2 + 1) + '.mrc')
    mt_df.insert(0, 'Tomogram', new_names)
    if out_df is None:
        out_df = mt_df.copy(deep=True)
    else:
        out_df = pd.concat([out_df, mt_df])

# Store the out CSV
out_df.to_csv(ROOT_PATH + '/synth_mbprot.csv')