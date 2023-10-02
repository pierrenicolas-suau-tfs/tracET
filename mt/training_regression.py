# =============================================================================================
# DeepFinder - a deep learning approach to localize macromolecules in cryo electron tomograms
# =============================================================================================
# Copyright (C) Inria,  Emmanuel Moebel, Charles Kervrann, All Rights Reserved, 2015-2021, v1.0
# License: GPL v3.0. See <https://www.gnu.org/licenses/>
# =============================================================================================
import h5py
import numpy as np
import time

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
import tensorflow.keras.backend as K

from sklearn.metrics import precision_recall_fscore_support

from deepfinder import models
from mt import losses
from deepfinder.utils import core
from deepfinder.utils import common as cm

from deepfinder.training import Train
from .models import deep_finder_regression

import scipy.ndimage
#from tensorflow.keras.losses import MeanSquaredError


# TODO: add method for resuming training. It should load existing weights and train_history. So when restarting, the plot curves show prececedent epochs
class TrainRegression(Train):
    # def __init__(self, Ncl, dim_in):
    #     Train.__init__(self, Ncl, dim_in)
    def __init__(self, dim_in):
            core.DeepFinder.__init__(self)
            self.path_out = './'


            # Network parameters:
            self.dim_in = dim_in  # /!\ has to a multiple of 4 (because of 2 pooling layers), so that dim_in=dim_out
            self.net = deep_finder_regression(self.dim_in)

            self.label_list = [0, 1] 
            
            # Training parameters:
            self.batch_size = 25
            self.epochs = 100
            self.steps_per_epoch = 100
            self.steps_per_valid = 10  # number of samples for validation
            self.optimizer = Adam(lr=0.00005, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
            self.loss = 'mean_squared_error' # losses.l2_norm

            self.flag_direct_read = 1
            self.flag_batch_bootstrap = 0
            self.Lrnd = 13  # random shifts applied when sampling data- and target-patches (in voxels)

            self.class_weight = None
            self.sample_weights = None  # np array same lenght as objl_train

            self.check_attributes()
       
    def check_attributes(self):
        #self.is_positive_int(self.Ncl, 'Ncl')
        self.is_multiple_4_int(self.dim_in, 'dim_in')
        self.is_positive_int(self.batch_size, 'batch_size')
        self.is_positive_int(self.epochs, 'epochs')
        self.is_positive_int(self.steps_per_epoch, 'steps_per_epoch')
        self.is_positive_int(self.steps_per_valid, 'steps_per_valid')
        self.is_int(self.Lrnd, 'Lrnd')
        
    # This function launches the training procedure. For each epoch, an image is plotted, displaying the progression
    # with different metrics: loss, accuracy, f1-score, recall, precision. Every 10 epochs, the current network weights
    # are saved.
    # INPUTS:
    #   path_data     : a list containing the paths to data files (i.e. tomograms)
    #   path_target   : a list containing the paths to target files (i.e. annotated volumes)
    #   objlist_train : list of dictionaries containing information about annotated objects (e.g. class, position)
    #                   In particular, the tomo_idx should correspond to the index of 'path_data' and 'path_target'.
    #                   See utils/objl.py for more info about object lists.
    #                   During training, these coordinates are used for guiding the patch sampling procedure.
    #   objlist_valid : same as 'objlist_train', but objects contained in this list are not used for training,
    #                   but for validation. It allows to monitor the training and check for over/under-fitting. Ideally,
    #                   the validation objects should originate from different tomograms than training objects.
    # The network is trained on small 3D patches (i.e. sub-volumes), sampled from the larger tomograms (due to memory
    # limitation). The patch sampling is not realized randomly, but is guided by the macromolecule coordinates contained
    # in so-called object lists (objlist).
    # Concerning the loading of the dataset, two options are possible:
    #    flag_direct_read=0: the whole dataset is loaded into memory
    #    flag_direct_read=1: only the patches are loaded into memory, each time a training batch is generated. This is
    #                        usefull when the dataset is too large to load into memory. However, the transfer speed
    #                        between the data server and the GPU host should be high enough, else the procedure becomes
    #                        very slow.
    # TODO: delete flag_direct_read. Launch should detect if direct_read is desired by checking if input data_list and
    #       target_list contain str (path) or numpy array
    def launch(self, path_data, path_target, objlist_train, objlist_valid):
        """This function launches the training procedure. For each epoch, an image is plotted, displaying the progression
        with different metrics: loss, accuracy, f1-score, recall, precision. Every 10 epochs, the current network weights
        are saved.

        Args:
            path_data (list of string): contains paths to data files (i.e. tomograms)
            path_target (list of string): contains paths to target files (i.e. annotated volumes)
            objlist_train (list of dictionaries): contains information about annotated objects (e.g. class, position)
                In particular, the tomo_idx should correspond to the index of 'path_data' and 'path_target'.
                See utils/objl.py for more info about object lists.
                During training, these coordinates are used for guiding the patch sampling procedure.
            objlist_valid (list of dictionaries): same as 'objlist_train', but objects contained in this list are not
                used for training, but for validation. It allows to monitor the training and check for over/under-fitting.
                Ideally, the validation objects should originate from different tomograms than training objects.

        Note:
            The function saves following files at regular intervals:
                net_weights_epoch*.h5: contains current network weights

                net_train_history.h5: contains arrays with all metrics per training iteration

                net_train_history_plot.png: plotted metric curves

        """
        self.check_attributes()
        self.check_arguments(path_data, path_target, objlist_train, objlist_valid)


        # Build network (not in constructor, else not possible to init model with weights from previous train round):
        # self.net.compile(optimizer=self.optimizer, loss=self.loss, metrics=['accuracy'])
        self.net.compile(optimizer=self.optimizer, loss=self.loss, metrics=['binary_accuracy'])

        # Load whole dataset:
        if self.flag_direct_read == False:
            self.display('Loading dataset ...')
            data_list, target_list = core.load_dataset(path_data, path_target, self.h5_dset_name)

        self.display('Launch training ...')

        # Declare lists for storing training statistics:
        hist_loss_train = []
        hist_acc_train  = []
        hist_loss_valid = []
        hist_acc_valid  = []
        hist_f1         = []
        hist_recall     = []
        hist_precision  = []
        process_time    = []

        # Training loop:
        for e in range(self.epochs):
            # TRAINING:
            start = time.time()
            list_loss_train = []
            list_acc_train = []
            for it in range(self.steps_per_epoch):
                if self.flag_direct_read:
                    batch_data, batch_target = self.generate_batch_direct_read(path_data, path_target, self.batch_size, objlist_train)
                else:
                    batch_data, batch_target, idx_list = self.generate_batch_from_array(data_list, target_list, self.batch_size, objlist_train)

                if self.sample_weights is not None:
                    sample_weight = self.sample_weights[idx_list]
                else:
                    sample_weight = None

                loss_train = self.net.train_on_batch(batch_data, batch_target,
                                                     class_weight=self.class_weight,
                                                     sample_weight=sample_weight)

                self.display('epoch %d/%d - it %d/%d - loss: %0.10f - acc: %0.3f' % (e + 1, self.epochs, it + 1, self.steps_per_epoch, loss_train[0], loss_train[1]))
                list_loss_train.append(loss_train[0])
                list_acc_train.append(loss_train[1])
            hist_loss_train.append(list_loss_train)
            hist_acc_train.append(list_acc_train)

            # VALIDATION (compute statistics to monitor training):
            list_loss_valid = []
            list_acc_valid  = []
            list_f1         = []
            list_recall     = []
            list_precision  = []
            for it in range(self.steps_per_valid):
                if self.flag_direct_read:
                    batch_data_valid, batch_target_valid = self.generate_batch_direct_read(path_data, path_target, self.batch_size, objlist_valid)
                else:
                    batch_data_valid, batch_target_valid, idx_list = self.generate_batch_from_array(data_list, target_list, self.batch_size, objlist_valid)
                loss_val = self.net.evaluate(batch_data_valid, batch_target_valid, verbose=0) # TODO replace by loss() to reduce computation
                batch_pred = self.net.predict(batch_data_valid)
                #loss_val = K.eval(losses.tversky_loss(K.constant(batch_target_valid), K.constant(batch_pred)))
                
                # scores = precision_recall_fscore_support(
                #     np.int8(batch_target_valid[:,:,:,:,0]<1).flatten(),
                #     np.int8(batch_pred[:,:,:,:,0]<1).flatten(),
                #     average=None,
                #     labels=self.label_list
                # )
                scores = precision_recall_fscore_support(
                    np.int8(batch_target_valid[:, :, :, :, 0] < 6).flatten(),
                    np.int8(batch_pred[:, :, :, :, 0] < 6).flatten(),
                    average=None,
                    labels=self.label_list
                )

                list_loss_valid.append(loss_val[0])
                list_acc_valid.append(loss_val[1])
                list_f1.append(scores[2])
                list_recall.append(scores[1])
                list_precision.append(scores[0])

            hist_loss_valid.append(list_loss_valid)
            hist_acc_valid.append(list_acc_valid)
            hist_f1.append(list_f1)
            hist_recall.append(list_recall)
            hist_precision.append(list_precision)

            end = time.time()
            process_time.append(end - start)
            self.display('-------------------------------------------------------------')
            self.display('EPOCH %d/%d - valid loss: %0.10f - valid acc: %0.3f - %0.2fsec' % (
            e + 1, self.epochs, loss_val[0], loss_val[1], end - start))


            # Save and plot training history:
            history = {'loss': hist_loss_train, 'acc': hist_acc_train, 'val_loss': hist_loss_valid,
                       'val_acc': hist_acc_valid, 'val_f1': hist_f1, 'val_recall': hist_recall,
                       'val_precision': hist_precision}
            core.save_history(history, self.path_out+'net_train_history.h5')
            core.plot_history(history, self.path_out+'net_train_history_plot.png')

            self.display('=============================================================')

            if (e + 1) % 10 == 0:  # save weights every 10 epochs
                self.net.save(self.path_out+'net_weights_epoch' + str(e + 1) + '.h5')

        self.display("Model took %0.2f seconds to train" % np.sum(process_time))
        self.net.save(self.path_out+'net_weights_FINAL.h5')


   
    # Generates batches for training and validation. In this version, the whole dataset has already been loaded into
    # memory, and batch is sampled from there. Apart from that does the same as above.
    # Is called when self.flag_direct_read=False
    # INPUTS:
    #   data: list of numpy arrays
    #   target: list of numpy arrays
    #   batch_size: int
    #   objlist: list of dictionnaries
    # OUTPUT:
    #   batch_data: numpy array [batch_idx, z, y, x, channel] in our case only 1 channel
    #   batch_target: numpy array [batch_idx, z, y, x, class_idx] is one-hot encoded
    def generate_batch_from_array(self, data, target, batch_size, objlist=None):
        p_in = int(np.floor(self.dim_in / 2))

        batch_data = np.zeros((batch_size, self.dim_in, self.dim_in, self.dim_in, 1))
        batch_target = np.zeros((batch_size, self.dim_in, self.dim_in, self.dim_in, 1))

        # The batch is generated by randomly sampling data patches.
        if self.flag_batch_bootstrap:  # choose from bootstrapped objlist
            pool = core.get_bootstrap_idx(objlist, Nbs=batch_size)
        else:  # choose from whole objlist
            pool = range(0, len(objlist))

        idx_list = []
        for i in range(batch_size):
            # choose random sample in training set:
            index = np.random.choice(pool)
            idx_list.append(index)

            tomoID = int(objlist[index]['tomo_idx'])

            tomodim = data[tomoID].shape

            sample_data = data[tomoID]
            sample_target = target[tomoID]

            # Get patch position:
            x, y, z = core.get_patch_position(tomodim, p_in, objlist[index], self.Lrnd)

            # Get patch:
            patch_data   = sample_data[  z-p_in:z+p_in, y-p_in:y+p_in, x-p_in:x+p_in]
            patch_target = sample_target[z-p_in:z+p_in, y-p_in:y+p_in, x-p_in:x+p_in]

            # Process the patches in order to be used by network:
            patch_data = (patch_data - np.mean(patch_data)) / np.std(patch_data)  # normalize
            
            #patch_target_onehot = to_categorical(patch_target, self.Ncl)

            # Store into batch array:
            batch_data[i, :, :, :, 0] = patch_data
            batch_target[i, :, :, :, 0] = patch_target

            # Data augmentation (180degree rotation around tilt axis):
            if np.random.uniform() < 0.5:
                batch_data[i] = np.rot90(batch_data[i], k=2, axes=(0, 2))
                batch_target[i] = np.rot90(batch_target[i], k=2, axes=(0, 2))

        return batch_data, batch_target, idx_list

