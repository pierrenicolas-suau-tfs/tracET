import os
import sys
import time
import nrrd
import getopt
import numpy as np

from tracET.core import lio
from tracET.core.skel import surface_skel, line_skel,point_skel
from tracET.core.diff import prepare_input,remove_borders,downsample_3d

def main():
    argv = sys.argv[1:]

    start=time.time()
    # Input parsing
    in_tomo,s=None,None
    skel_mode=None
    ibin=None
    f=None
    downsample=None

    try:
        opts, args = getopt.getopt(argv, "hi:s:m:b:f:d:",["help","itomo","sdesv","smode","ibin","filt","downs"])
    except getopt.GetoptError:
        print('python apply_nonmaxsup_pre.py -i <in_tomo> -s <smooth_desv> -m <skel_mode> -b <binary_input> -f <filter> -d <downsample>')
        sys.exit()
    for opt, arg in opts:
        if opt in ("-h","--help"):
            print('python trace_graph.py -i <in_tomo> -r <radius> -s <subsampling> -o <out_dir>')
            print('\t-i (--itomo) <in_tomo> input tomogram')
            print('\t-s (--sdesv) <smooth_desviation> desviation for gaussian filter (~1/3 tubule radium)')
            print('\t-m (--mode) <skel_mode> structural mode for computing the skeleton: '
                  '\'s\' surface, \'l\' line and \'b\' blob ')
            print('\t-b (--ibin) <binary_input> 1 if the input is a binary map, it create a distance transformation'
                  'at the begining. (optional, default 0, a scalar map)')
            print('\t-f (--filt) <filter> filter for the mask (optional, default 0)')
            print('\t-d (--downs) <downsample> radius in voxels to downsample the skeletons. (optional,default is 0)')

            sys.exit()
        elif opt in ("-i","--itomo"):
            in_tomo=arg
            if not(os.path.splitext(in_tomo)[1] in ('.mrc', '.nhdr', '.nrrd')):
                print('The input file must have a .mrc, .nhdr or nrrd extension!')
                sys.exit()
        elif opt in ("-s","--sdesv"):
            s=arg

        elif opt in ("-m", "smode"):
            skel_mode = arg
            if not (skel_mode in ('s', 'l', 'b')):
                print('The input evec must be "s" (surface), "l" (linear) or "b" (blob)!')
                sys.exit()
        elif opt in ("-b","--ibin"):
            ibin=bool(int(arg))
        elif opt in ("-f","--filt"):
            f = float(arg)
        elif opt in ("-d","--downs"):
            downsample=float(arg)

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

    if skel_mode is not None:
        print('Structure to analysis',skel_mode)
    else:
        print('python apply_nonmaxsup_pre.py -i <in_tomo> -s <smooth_desv> -e <eigenvalues> -m <skel_mode> -b <binary_input> -f <filter> -o <out_tomo>')
        print('Almost an input sdesv parameter -s must be provided!')
        sys.exit()
    if ibin is not None:
        print('Binary mode: ',str(ibin))
    else:
        ibin=False
        print('Default: Binary mode ',str(ibin))
    if f is not None:
        print ('Trheshold filter in the masc is ',str(f))
    else:
        f=0
        print('Default: Trheshold filter in the masc is ', str(f))
    if downsample is not None:
        print('downsampled applied with ',str(downsample),' voxels')
    else:
        downsample=0
        print('Default. Not downsampled')


    T=prepare_input(T,sigma=s,bin=ibin,imf=None)
    sal_time=time.time()
    print('The saliency map lasted ', str(sal_time - start), ' s in execute')
    if skel_mode == 's':
        P = surface_skel(T,f)
    elif skel_mode == 'l':
        P = line_skel(T,f)

    else:
        P = point_skel(T,f)

    P = remove_borders(P)
    P = downsample_3d(P,skel_dsample=downsample)
    print('Saving')
    if os.path.splitext(in_tomo)[1] == '.mrc':
        lio.write_mrc(P.astype(np.float32), os.path.splitext(in_tomo)[0] + '_supred.mrc')
    else:
        nrrd.write(os.path.splitext(in_tomo)[0]+'supred.nrrd', P)
    end = time.time()
    print ('The nms lasted ',str(end-sal_time),' s in execute')
    print('Successfully terminated. (' + time.strftime("%c") + ')')


if __name__ == "__main__":
    main()

