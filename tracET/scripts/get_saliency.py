import os
import sys
import time
import nrrd
import getopt
import scipy
import numpy as np

from tracET.core import lio

from tracET.core.diff import prepare_input

def main():
    argv = sys.argv[1:]
    start=time.time()
    # Input parsing
    in_tomo,s=None,None

    try:
        opts, args = getopt.getopt(argv, "hi:s:",["help","itomo","sdesv"])
    except getopt.GetoptError:
        print('python get_saliency.py -i <in_tomo> -s <smooth_desv>')
        sys.exit()
    for opt, arg in opts:
        if opt in ("-h","--help"):
            print('python trace_graph.py -i <in_tomo> -r <radius> -s <subsampling> -o <out_dir>')
            print('\t-i (--itomo) <in_tomo> input tomogram')
            print('\t-s (--sdesv) <smooth_desviation> desviation for gaussian filter (~1/3 tubule radium)')


            sys.exit()
        elif opt in ("-i","--itomo"):
            in_tomo=arg
            if not(os.path.splitext(in_tomo)[1] in ('.mrc', '.nhdr', '.nrrd')):
                print('The input file must have a .mrc, .nhdr or nrrd extension!')
                sys.exit()
        elif opt in ("-s","--sdesv"):
            s=arg

        else:
            print('python apply_nonmaxsup_pre.py -i <in_tomo> -s <smooth_desv> -e <eigenvalues> -v <eigenvectors> -f <filter> -o <out_tomo>')
            print("Inputs must be one of them!")
            sys.exit()

    if in_tomo is not None:
        print('\t-Loading input tomogram:', in_tomo)
        if os.path.splitext(in_tomo)[1] == '.mrc':
            T = lio.load_mrc(in_tomo)
        else:
            T = nrrd.read(in_tomo)[0]
    else:
        print('python apply_nonmaxsup_pre.py -i <in_tomo> -s <smooth_desv> -e <eigenvalues> -m <skel_mode> -b <binary_input> -f <filter> -o <out_tomo>')
        print('Almost an input tomogram -i must be provided!')
        sys.exit()
    if s is not None:
        s=float(s)
        print('Smooth desviation: ',str(s))
    else:
        prin('python apply_nonmaxsup_pre.py -i <in_tomo> -s <smooth_desv> -e <eigenvalues> -m <skel_mode> -b <binary_input> -f <filter> -o <out_tomo>')
        print('Almost an input sdesv parameter -s must be provided!')
        sys.exit()




    P=prepare_input(T,sigma=s,bin=True,imf=None)
    tomo_dsts = scipy.ndimage.morphology.distance_transform_edt(T)
    print('Saving')
    if os.path.splitext(in_tomo)[1] == '.mrc':
        lio.write_mrc(P.astype(np.float32), os.path.splitext(in_tomo)[0] + '_saliency.mrc')
        lio.write_mrc(tomo_dsts.astype(np.float32), os.path.splitext(in_tomo)[0] + '_distance.mrc')
    else:
        nrrd.write(os.path.splitext(in_tomo)[0] + '_saliency.nrrd', P)
    sal_time=time.time()
    print('The saliency map lasted ', str(sal_time - start), ' s in execute')

    print('Successfully terminated. (' + time.strftime("%c") + ')')


if __name__ == "__main__":
    main()