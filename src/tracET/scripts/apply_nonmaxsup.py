import os
import sys
import time
#import nrrd
import getopt

import numpy as np

from src.tracET.core import lio
from src.tracET.core.skel import surface_skel, line_skel,point_skel
from src.tracET.core.diff import prepare_input,remove_borders

def main(argv):
    # Input parsing
    in_tomo,s=None,None
    eval,skel_mode=None,None
    ibin=None
    f=None
    out_tomo=None
    try:
        opts, args = getopt.getopt(argv, "hi:s:e:m:b:f:o:",["help","itomo","sdesv","eval","smode","ibin","filt","otomo"])
    except getopt.GetoptError:
        print('python apply_nonmaxsup_pre.py -i <in_tomo> -s <smooth_desv> -e <eigenvalues> -m <skel_mode> -b <binary_input> -f <filter> -o <out_tomo>')
        sys.exit()
    for opt, arg in opts:
        if opt in ("-h","--help"):
            print('python trace_graph.py -i <in_tomo> -r <radius> -s <subsampling> -o <out_dir>')
            print('\t-i (--itomo) <in_tomo> input tomogram')
            print('\t-s (--sdesv) <smooth_desviation> desviation for gaussian filter (~1/3 tubule radium')
            print('\t-e (--eval) <eigenvalues> type of eigenvalues and eigenvectors "Hessian" or "Struct"'
                  ' (optional, default "Hessian")')
            print('\t-m (--mode) <skel_mode> structural mode for computing the skeleton: '
                  '\'s\' surface, \'l\' line and \'b\' blob ')
            print('\t-b (--ibin) <binary input> True if the input is a binary map, it create a distance transformation'
                  'at the begining. (optional, default False)')
            print('\t-f (--filt) <filter> filter for the mask (optional, default 1)')
            print('\t-o (--otomo) <out_tomo> output binary tomogram (point cloud)')
            sys.exit()
        elif opt in ("-i","--itomo"):
            in_tomo=arg
            if not(os.path.splitext(in_tomo)[1] in ('.mrc', '.nhdr', '.nrrd')):
                print('The input file must have a .mrc, .nhdr or nrrd extension!')
                sys.exit()
        elif opt in ("-s","--sdesv"):
            s=arg
        elif opt in ("-e","eval"):
            eval=arg
            if not (eval in ('Hessian','Struct')):
                print('The input eval must be "hessian" or "struct"!')
                sys.exit()
        elif opt in ("-m", "smode"):
            skel_mode = arg
            if not (skel_mode in ('s', 'l', 'b')):
                print('The input evec must be "s" (surface), "l" (linear) or "b" (blob)!')
                sys.exit()
        elif opt in ("-b","--ibin"):
            ibin=bool(arg)
        elif opt in ("-f","--filt"):
            f = float(arg)
        elif opt in ("-o","--otomo"):
            out_tomo = arg
            if not(os.path.splitext(out_tomo)[1] in ('.mrc', '.nhdr', '.nrrd')):
                print('The output file must have a .mrc, .nhdr or nrrd extension!')
                sys.exit()
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
    if eval is not None:
        print('Use ',eval, ' eigenvalues')
    else:
        eval="hessian"
        print('Default: Use ', eval, ' eigenvalues')
    if skel_mode is not None:
        print('Structure to analysis',skel_mode)
    else:
        print('python apply_nonmaxsup_pre.py -i <in_tomo> -s <smooth_desv> -e <eigenvalues> -m <skel_mode> -b <binary_input> -f <filter> -o <out_tomo>')
        print('Almost an input sdesv parameter -s must be provided!')
        sys.exit()
    if f is not None:
        print ('Trheshold filter in the masc is ',str(f))
    else:
        f=5
        print('Default: Trheshold filter in the masc is ', str(f))
    if out_tomo is not None:
        print ('Output tomogram: ',out_tomo)
    else:
        print('python apply_nonmaxsup_pre.py -i <in_tomo> -s <smooth_desv> -e <eigenvalues> -v <eigenvectors> -f <filter> -o <out_tomo>')
        print('Almost an output tomogram -o must be provided!')
        sys.exit()

    T=prepare_input(T,sigma=s,bin=ibin,imf=None)
    if skel_mode == 's':
        P = surface_skel(T,f)
    elif skel_mode == 'l':
        P = line_skel(T,f,mode=eval)

    else:
        P = point_skel(T,f,mode=eval)

    P = remove_borders(P)
    print('Saving')
    if os.path.splitext(out_tomo)[1] == '.mrc':
        lio.write_mrc(P.astype(np.float32), out_tomo)
    else:
        nrrd.write(out_tomo, P)

    print('Successfully terminated. (' + time.strftime("%c") + ')')


if __name__ == "__main__":
    main(sys.argv[1:])

