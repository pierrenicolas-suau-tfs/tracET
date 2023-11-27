import os
import random

import math
import numpy
import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt


from mt.representation import points_to_btomo, seg_dist_trans, gauss_decay
from core.lio import load_mrc, write_mrc
#import deepfinder.utils.objl as ol

ROOT_DIR= '/data/chiem/pelayo/neural_network/'
csv_file='Aproximation.csv'
tomogram = 'Ctrl_20220511_368d_tomo06_tubule'
mt_dst=30
zf=1
tubules_df=pd.read_csv(ROOT_DIR+csv_file, sep='\t')

tomogram_df=tubules_df[(tubules_df["Tomogram"]==tomogram)]

coords=tomogram_df[["XCoord","YCoord","ZCoord"]].to_numpy()
coords=np.round(coords).astype(int)

map=np.zeros((1024,1440,464),dtype=bool)
points_to_btomo(coords, map, True)
tomo_dst = seg_dist_trans(map)
tomo_mt = tomo_dst <= mt_dst * zf
tomo_mt_dst = tomo_dst - mt_dst
tomo_mt_dst[tomo_mt_dst > 0] = 0
tomo_mt_dst *= -1
out_tomo=np.zeros((1024,1440,464))
out_tomo[tomo_mt_dst>0]=1
write_mrc(out_tomo[:1023,:,:].astype(np.float32),'/data/chiem/pelayo/neural_network/Ctrl_20220511_368d_tomo06_dist_map.mrc')
tomo_prueba=load_mrc(ROOT_DIR+'Ctrl_20220511_368d_tomo06_reg_synth.mrc')
tomo_out2=np.zeros((1023,1440,464))
tomo_out2[tomo_prueba>4]=1
write_mrc(tomo_out2.astype(np.float32),'/data/chiem/pelayo/neural_network/try_dice/Ctrl_20220511_368d_tomo06_detection_try.mrc')