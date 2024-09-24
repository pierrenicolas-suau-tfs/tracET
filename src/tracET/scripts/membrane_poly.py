import sys, os, getopt, time
from src.tracET.core.vtk_uts import *
from src.tracET.core import lio
from src.tracET.representation.graphs import *
from src.tracET.representation.curve import *
import tracemalloc
#import nrrd
import pandas as pd

from sklearn.cluster import DBSCAN
import open3d as o3d




#tomo = '/project/chiem/pelayo/neural_network/nonmaxsup_outputs/tomo_jenny/lam8/membrane/tomo01_lam8_deconv_crop_seg_membrane_supred_time.mrc'
#epsilon = 10
#min_samples = 2

def main(argv):
    start = time.time()
    tomo = None
    epsilon = None
    min_samples = None
    try:
        opts, args = getopt.getopt(argv,"hi:d:s:",["help","itomo","dist","samp"])
    except getopt.GetoptError:
        print('python membrane_poly.py -i <in_tomo> -d <distance_clustering> -s <min_samples>')
    for opt, arg in opts:
        if opt in ("-h","--help"):
            print('python membrane_poly.py -i <in_tomo> -d <distance_clustering> -s <min_samples>')
            print('\t-i (--itomo) <in_tomo> input tomogram (point cloud) of membranes.')
            print('\t-d (--dist) <distance_clustering> Distance of points to be part of the same cluster')
            print('\t-s (--samp) <min_samples> Minimum samples to make a cluster. (Optional) Default: 2.')
        elif opt in ("-i", "--itomo"):
            tomo = arg
        if not (os.path.splitext(tomo)[1] in ('.mrc', '.nhdr', '.nrrd')):
            print('The input file must have a .mrc, .nhdr or nrrd extension!')
            sys.exit()
        elif opt in ("-d", "--dist"):
            epsilon = np.float32(arg)
        elif opt in ("-s","--samp"):
            min_samples = np.int32(arg)
    if tomo is not None:
        print('\t-Loading input tomogram:', tomo)
        if os.path.splitext(tomo)[1] == '.mrc':
            T = lio.load_mrc(tomo)
        else:
            T = nrrd.read(tomo)[0]
    else:
        print('python membrane_poly.py -i <in_tomo> -d <distance_clustering> -s <min_samples>')
        print('Almost an input tomogram -i must be provided')
        sys.exit()
    if epsilon is not None:
        print('\t-Distance clustering: ', str(epsilon))
    else:
        print('python membrane_poly.py -i <in_tomo> -d <distance_clustering> -s <min_samples>')
        print('Almost an input tomogram -d must be provided')
        sys.exit()
    if min_samples is not None:
        print('\t-Minimum number of samples for cluster: ',str(min_samples))
    else:
        min_samples = np.int32(2)
        print('\t-Default: Minimum number of samples for cluster: 2')







    coords=np.argwhere(T)
    db = DBSCAN(eps=epsilon,min_samples=min_samples).fit(coords)
    labels = db.labels_
    pos_0 = np.where(labels==0)[0]
    surf_0 = coords[pos_0]


    poly = points_to_poly(surf_0)

    add_label_to_poly(poly,0,'Membrane')


    for label in range(1,max(labels)):
        pos= np.where(labels==label)[0]
        surf = coords[pos]


        poly_i = points_to_poly(surf)
        add_label_to_poly(poly_i, label, 'Membrane')


        poly = merge_polys(poly,poly_i)



    save_vtp(poly, os.path.splitext(tomo)[0] + '.vtp')
    end = time.time()
    print('lasted '+str(end-start)+' s')


if __name__ == "__main__":
    main(sys.argv[1:])