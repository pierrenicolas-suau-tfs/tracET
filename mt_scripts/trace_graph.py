
import sys, getopt, time
from vtk_uts import *
from graph_uts import *
from core import lio
import pandas as pd

def main(argv):
    #Input parsing
    main_dir = None
    in_tomo, out_dir = None, None
    r, s = None, None
    try:
        opts, args = getopt.getopt(argv, "hm:i:r:s:o:",["help","main","itomo","rad","subsam","odir"])
    except getopt.GetoptError:
        print('python trace_graph.py -i <in_tomo> -r <radius> -s <subsampling> -o <out_dir>')
        sys.exit()
    for opt, arg in opts:
        if opt in ("-h","--help"):
            print('python trace_graph.py - m <main_dir> -i <in_tomo> -r <radius> -s <subsampling> -o <out_dir>')
            print('\t-m (--main) main directory ')
            print('\t-i (--itomo) <in_tomo> input tomogram (point cloud)')
            print('\t-r (--rad) <radius> radius to connect points in the graph')
            print('\t-s (--subsam) <subsampling> radius of subsampling (optional, default no subsampling)')
            print('\t-o (--odir) <out_dir> putput directory')
        elif opt in ("-m","--main"):
            main_dir = arg
        elif opt in ("-i","--itomo"):
            in_tomo=arg
            if not(os.path.splitext(in_tomo)[1] in ('.mrc', '.nhdr', '.nrrd')):
                print('The input file must have a .mrc, .nhdr or nrrd extension!')
                sys.exit()
        elif opt in ("-r","--rad"):
            r=arg
        elif opt in ("-s","--subsam"):
            s=arg
        elif opt in ("-o","--odir"):
            out_dir=arg

    if main_dir is not None:
        print('\t-Main directory:', main_dir)
    else:
        print('python trace_graph.py - m <main_dir> -i <in_tomo> -r <radius> -s <subsampling> -o <out_dir>')
        print('Almost a main directory -m must be provided')
    if in_tomo is not None:
        print('\t-Loading input tomogram:', in_tomo)
        if os.path.splitext(in_tomo)[1] == '.mrc':
            T = lio.load_mrc(main_dir+in_tomo)
        else:
            T = nrrd.read(main_dir+in_tomo)[0]
    else:
        print('python trace_graph.py -i <in_tomo> -r <radius> -s <subsampling> -o <out_dir>')
        print('Almost an input tomogram -i must be provided')
        sys.exit()
    if r is not None:
        print('Radius = ',str(r))
    else:
        print('python trace_graph.py -i <in_tomo> -r <radius> -s <subsampling> -o <out_dir>')
        print('Almost an input radius -r must be provided')
        sys.exit()
    if s is not None:
        print ('Subsampling radius = ',str(s))
    else:
        s=0
        print('Default no subsampling')
    if out_dir is not None:
        print ('Save out vtp in: ',out_dir)
    else:
        print('python trace_graph.py -i <in_tomo> -r <radius> -s <subsampling> -o <out_dir>')
        print('Almost an output directory -o must be provided')
        sys.exit()


    print('Calculating the graph')
    [coords, graph_array] =make_skeleton_graph(T,float(r),float(s))
    print('Spliting in components')
    [graph_ar_comps,coords_comps]=split_into_components(graph_array,coords)
    print('Making graphs line like and save')
    tubule_list=np.zeros(len(coords))
    a=0
    for i in range(len(graph_ar_comps)):
        print('Procesing tubule ',str(i))
        print('Removing cycles')
        L_graph=remove_cycles(graph_ar_comps[i])
        L_graph=remove_cycles(L_graph)
        print('Make polydata')
        Targets, Sources = (L_graph.nonzero())
        points_poly = make_graph_polydata(coords_comps[i],Sources,Targets)
        print('Saving')
        save_vtp(points_poly, main_dir+ out_dir + os.path.splitext(in_tomo)[0]+ '_skel_graph_tubule_'+str(i)+'.vtp')
        print(os.path.splitext(in_tomo)[0]+ '_skel_graph_tubule_'+str(i)+'.vtp'+' saved in '+main_dir+out_dir)
        tubule_list[a:a+len(coords_comps[i])]=i*np.ones(len(coords_comps[i]))
        a=a+len(coords_comps[i])
    out_mat=np.zeros((len(coords),4))
    out_mat[:,0]=tubule_list
    out_mat[:,1:4]=coords
    out_pd=pd.DataFrame(data=out_mat,columns=['Filament','X','Y','Z'])
    out_pd.to_csv(main_dir+os.path.splitext(in_tomo)[0]+ '_skel_graph.csv')

    print('Successfully terminated. (' + time.strftime("%c") + ')')

if __name__ == "__main__":
    main(sys.argv[1:])