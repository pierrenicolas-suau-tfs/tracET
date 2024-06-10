"""
Script for computing the skeleton DICE to measure overlapping between two segmented tomograms
    - Input:
        + The two tomograms to compared
            * Input tomo
            * Ground truth tomo (reference)
        + Sekeleton structure
            * Surface (e.g. membranes)
            * Line (e.g. actin filaments and microtubules)
            * Blobs (e.g. globular macromolecules)
    - Output:
        + DICE scores (computed as described in https://doi.org/10.1109/CVPR46437.2021.01629):
            * DICE metric
            * TP (Topological Precision)
            * TS (Topological Sensitivity)
        + (Optional) the skeleton generated for computing the metrics
"""

import os
import sys
import time
#import nrrd
import getopt

import numpy as np

from src.tracET.core import lio
from src.tracET.metrics.dice2 import cs_dice, cl_dice, pt_dice


def print_help_msg():
    """
    Print help message
    :return:
    """
    print('python', os.path.basename(__file__), '-i <in_tomo> -g <gt_tomo> -m <skel_mode>')
    print('\t-i (--itomo) <in_tomo> input tomogram')
    print('\t-g (--igt) <gt_tomo> ground truth tomogram')
    print('\t-m (--mode) <skel_mode> structural mode for computing the skeleton: \'s\' surface, \'l\' line and \'b\' blob ')
    print('\t-o (--otomo) <out_tomo_skel> (optional) path to store the skeleton generated of the tomogram')
    print('\t-t (--ogt) <out_gt_skel> (optional) path to store the skeleton generated of the ground truth')
    print('\t-d (--dil) <dilaton> (optional) number of iterations to pre-dilate (make thicker) the input segmantations')
    print('\t-f (--ifilt)  <ifilter> filter for the mask for the input tomogram. (optional, default 1)')
    print('\t-F (--tfilt)  <ifilter> filter for the mask for the ground truth tomogram.(optional, default 1)')


def main(argv):
    start = time.time()
    # Input parsing
    in_tomo, in_tomo_gt = None, None
    skel_mode = None
    out_tomo_skel, out_tomo_gt_skel = None, None
    it_dil = None
    ibin, tbin = None, None
    ifilt, tfilt= None, None
    try:
        opts, args = getopt.getopt(argv, "hi:g:m:o:t:d:b:B:f:F:",["help", "itomo", "igt", "mode", "otomo",
                                                                           "ogt", "dil", "ibin", "tbin", "ifilt", "tfilt"])
    except getopt.GetoptError:
        print_help_msg()
        sys.exit()
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            sys.exit()
        elif opt in ("-i", "--itomo"):
            in_tomo = arg
            if not(os.path.splitext(in_tomo)[1] in ('.mrc', '.nhdr', '.nrrd')):
                print('The input file must have a .mrc, .nhdr or nrrd extension!')
                sys.exit()
        elif opt in ("-g", "--igt"):
            in_tomo_gt = arg
            if not (os.path.splitext(in_tomo_gt)[1] in ('.mrc', '.nhdr', '.nrrd')):
                print('The input file must have a .mrc, .nhdr or nrrd extension!')
                sys.exit()
        elif opt in ("-m", "mode"):
            skel_mode = arg
            if (skel_mode != 's') and (skel_mode != 'l') and (skel_mode != 'b'):
                print('The argument for parameter \'m\' (mode) must be one of: \'s\', \'l\' or \'b\'')
                sys.exit()
        elif opt in ("-o", "--otomo"):
            out_tomo_skel = arg
            if not (os.path.splitext(out_tomo_skel)[1] in ('.mrc', '.nhdr', '.nrrd')):
                print('The output file for the tomogram must have a .mrc, .nhdr or nrrd extension!')
                sys.exit()
        elif opt in ("-t", "--ogt"):
            out_tomo_gt_skel = arg
            if not (os.path.splitext(out_tomo_gt_skel)[1] in ('.mrc', '.nhdr', '.nrrd')):
                print('The output file for the ground truth must have a .mrc, .nhdr or nrrd extension!')
                sys.exit()
        elif opt in ("-d", "--dil"):
            it_dil = int(arg)
            if it_dil >= 0:
                print('The number of iterations for dilation must be greater or equal to zero!')
                sys.exit()
        elif opt in ("-b", "--ibin"):
            ibin = bool(eval(arg))
        elif opt in ("-B", "--tbin"):
            tbin = bool(eval(arg))
        elif opt in ("-f", "--ifilt"):
            ifilt = eval(arg)
        elif opt in ("-F", "--tfilt"):
            tfilt = eval(arg)
        else:
            print('The option \'' + opt + '\' is not recognized!')
            print_help_msg()
            sys.exit()

    # Loading the input tomograms
    if in_tomo is not None:
        print('\t-Loading input tomogram:', in_tomo)
        if os.path.splitext(in_tomo)[1] == '.mrc':
            tomo = lio.load_mrc(in_tomo).astype(np.float32)
        else:
            tomo = nrrd.read(in_tomo)[0].astype(np.float32)
    else:
        print('The input tomogram \'-i\' (--itomo) must be provided')
        print_help_msg()
        sys.exit()

    # Loading the ground tru
    # th tomograms
    if in_tomo is not None:
        print('\t-Loading the ground truth tomogram:', in_tomo_gt)
        if os.path.splitext(in_tomo_gt)[1] == '.mrc':
            tomo_gt = lio.load_mrc(in_tomo_gt).astype(np.float32)
        else:
            tomo_gt = nrrd.read(in_tomo_gt)[0].astype(np.float32)
    else:
        print('The ground truth tomogram \'-g\' (--igt) must be provided')
        print_help_msg()
        sys.exit()
    if it_dil is None:
        it_dil=0

    if ibin is None:
        ibin=True
    if tbin is None:
        tbin=True


    # Compute the appropriate metric
    if skel_mode == 's':
        results = cs_dice(tomo, tomo_gt, dilation=it_dil, tomo_bin=ibin, tomo_imf=ifilt, tomo_gt_bin=tbin, gt_imf=tfilt)
    elif skel_mode == 'l':
        results = cl_dice(tomo, tomo_gt, dilation=it_dil, tomo_bin=ibin, tomo_imf=ifilt, tomo_gt_bin=tbin, gt_imf=tfilt)
    elif skel_mode == 'b':
        results = pt_dice(tomo, tomo_gt, dilation=it_dil, tomo_bin=ibin, tomo_imf=ifilt, tomo_gt_bin=tbin, gt_imf=tfilt)
    else:
        print('Mode \'' + skel_mode + '\' not implemented!')

    # Print the output results
    print('\t-RESULTS:')
    print('\t\t+DICE:', results[0])
    print('\t\t+TP:', results[1])
    print('\t\t+TS:', results[2])

    # Storing the generated skeleton if needed
    if out_tomo_skel is not None:
        print('Storing the tomogram skeleton in the file:', out_tomo_skel)
        if os.path.splitext(out_tomo_skel)[1] == '.mrc':
            lio.write_mrc(results[3].astype(np.int8), out_tomo_skel)
        else:
            nrrd.write(out_tomo_skel, results[3].astype(np.int8))
    if out_tomo_gt_skel is not None:
        print('Storing the ground truth skeleton in the file:', out_tomo_gt_skel)
        if os.path.splitext(out_tomo_gt_skel)[1] == '.mrc':
            lio.write_mrc(results[4].astype(np.int8), out_tomo_gt_skel)
        else:
            nrrd.write(out_tomo_gt_skel, results[4].astype(np.int8))

    end = time.time()
    print('The program lasted ', str(end - start), ' s in execute')
    print('Successfully terminated. (' + time.strftime("%c") + ')')


if __name__ == "__main__":
    main(sys.argv[1:])
