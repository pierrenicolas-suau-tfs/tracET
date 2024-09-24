import sys, os, getopt, time
#import nrrd
from src.tracET.core import lio
from src.tracET.representation.clustering import *
from src.tracET.core.vtk_uts import *
import pandas as pd


def main(argv):
    start = time.time()

    #Input parsing
    in_tomo = None
    mode = None
    blob_size = None
    n_jobs = None
    try:
        opts, args = getopt.getopt(argv,"hi:m:b:n:",["help","itomo","mode","blob_d","n_jobs"])
    except getopt.GetoptError:
        print('python Get_cluster.py -i <in_tomo> -b <blob_diameter> -n <n_jobs>')
    for opt, arg in opts:
        if opt in ("-h","--help"):
            print('python Get_cluster.py -i <in_tomo> -b <blob_diameter> -n <n_jobs>')
            print('\t-i (--itomo) <in_tomo> input tomogram (point cloud) of blobs.')
            print('\t-m (--mode) <mode> "Affinity" or "MeanShift". Algorithm of clustering')
            print('\t-b (--blob_d) <blob_diameter> If Mode = MeanShift: Diameter of the blobs to detect.')
            print('\t-n (--n_jobs) <n_jobs> (Optional) Number of jobs for execute the clustering. (Default: One job)')
        elif opt in ("-i", "--itomo"):
            in_tomo = arg
            if not (os.path.splitext(in_tomo)[1] in ('.mrc', '.nhdr', '.nrrd')):
                print('The input file must have a .mrc, .nhdr or nrrd extension!')
                sys.exit()
        elif opt in("-m","--mode"):
            mode= arg
            if not mode in ('Affinity','MeanShift'):
                print('Mode should be "Affinity" or "MeanShift".')
                sys.exit()
        elif opt in ("-b","--blob_d"):
            blob_size=arg
        elif opt in ("-n","--n_jobs"):
            n_jobs=arg
    if in_tomo is not None:
        print('\t-Loading input tomogram:', in_tomo)
        if os.path.splitext(in_tomo)[1] == '.mrc':
            T = lio.load_mrc(in_tomo)
        else:
            T = nrrd.read(in_tomo)[0]
    else:
        print('python Get_cluster.py -i <in_tomo> -b <blob_diameter> -n <n_jobs>')
        print('Almost an input tomogram -i must be provided')
        sys.exit()
    if mode is not None:
        print('Clustering by ', mode, 'Method')
    else:
        print('python Get_cluster.py -i <in_tomo> -b <blob_diameter> -n <n_jobs>')
        print('Almost an input mode -m must be provided')
    if mode == 'MeanShift':
        if blob_size is not None:
            blob_size=int(blob_size)
            print('\t-Size of the blobs:', str(blob_size))
        else:
            print('python Get_cluster.py -i <in_tomo> -b <blob_diameter> -n <n_jobs>')
            print('Almost the diameter of the blobs -b must be provided')
            sys.exit()
        if n_jobs is not None:
            n_jobs = int(n_jobs)
            print('\t-Jobs for calculate: ', str(n_jobs))
        else:
            print('\t-Jobs to calculate: 1 by default')

    if mode == 'Affinity':
        labels, centers, out_T, out_poly= get_AF_cluster(T)
    else:
        labels, centers, out_T, out_poly= get_MS_cluster(T,blob_size,n_jobs)



    save_vtp(out_poly, os.path.splitext(in_tomo)[0]+'_'+mode+'_labeled.vtp')
    lio.write_mrc(out_T.astype(np.float32), os.path.splitext(in_tomo)[0]+'_'+mode+'_labeled.mrc')

    mod_file_df=pd.DataFrame({'object': np.ones(len(centers[:,0])).astype(np.int32),'contourns':np.arange(len(centers[:,0]))+1,'X': centers[:,0].astype(np.int32),'Y': centers[:,1].astype(np.int32),'Z': centers[:,2].astype(np.int32)})

    mod_file_df.to_csv(os.path.splitext(in_tomo)[0]+'_'+mode+'_centers.txt',sep='\t',header=False,index=False)
    end = time.time()
    print('The program lasted ', str(end - start), ' s in execute')
    print('Successfully terminated. (' + time.strftime("%c") + ')')

if __name__ == "__main__":
    main(sys.argv[1:])

