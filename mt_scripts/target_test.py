import sys

# sys.path.append('/net/serpico-fs2/emoebel/github/deep-finder/')
# sys.path.append('/Users/emoebel/serpico-fs2/github/deep-finder')

import numpy as np
from deepfinder.training import Train
from deepfinder.utils import objl as ol
from deepfinder.utils import common as cm

prefix_data = '/media/martinez/Sistema/Users/Antonio/workspace/pycharm_proj/shrec_data/shrec21_full_dataset'
prefix_target = '/media/martinez/Sistema/Users/Antonio/workspace/pycharm_proj/shrec_data/deep-finder/out_v1/'

data = [
    prefix_data + '/model_0/reconstruction.mrc',
    prefix_data + '/model_1/reconstruction.mrc',
]
target = [
    prefix_target + 'target_tomo0.mrc',
    prefix_target + 'target_tomo1.mrc',
]

objl_train = ol.read_xml('/media/martinez/Sistema/Users/Antonio/workspace/pycharm_proj/shrec_data/deep-finder/in/particle_locations_models_0-1.xml')

Nclass = 13
dim_in = 56  # patch size

trainer = Train(Ncl=Nclass, dim_in=dim_in)
trainer.dim_in = 56  # patch size
trainer.batch_size = 2
trainer.Lrnd = 0

# ================================================================================================================
# Get macromolecules of class 2
objl_class = ol.get_class(objl_train, 2)
batch_data, batch_target, _ = trainer.generate_batch_from_array(data, target, trainer.batch_size, objlist=objl_class)

# Print the sampled batch as orthoslices:
for p in range(batch_target.shape[0]):
    patch_data = batch_data[p, :, :, :, 0]
    patch_target = np.int8(np.argmax(batch_target[p], axis=-1))
    cm.plot_volume_orthoslices(patch_data, 'class2_' + str(p) + '_data.png')
    cm.plot_volume_orthoslices(patch_target, 'class2_' + str(p) + '_target.png')

# ================================================================================================================
# Get macromolecules of class 13
objl_class = ol.get_class(objl_train, 12)
batch_data, batch_target, _ = trainer.generate_batch_from_array(data, target, trainer.batch_size, objlist=objl_class)

# Print the sampled batch as orthoslices:
for p in range(batch_target.shape[0]):
    patch_data = batch_data[p, :, :, :, 0]
    patch_target = np.int8(np.argmax(batch_target[p], axis=-1))
    cm.plot_volume_orthoslices(patch_data, 'class13_' + str(p) + '_data.png')
    cm.plot_volume_orthoslices(patch_target, 'class13_' + str(p) + '_target.png')
