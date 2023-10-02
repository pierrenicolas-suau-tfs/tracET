
import sys, getopt, time
from vtk_uts import *
from graph_uts import *
from mt import lio


def main(argv):
    #Input parsing
    in_tomo, out_dir = None, None
    r, s = None, None
    try:
        opts, args = getopt.getopt(argv, "hi:r:s:o:",["help","itomo","rad","subsam","odir"])
    except getopt.GetoptError:
        print('python trace_graph.py -i <in_tomo> -r <radius> -s <subsampling> -o <out_dir>')
        sys.exit()
    for opt, arg in opts:
        if opt in ("-h","--help"):
            print('python trace_graph.py -i <in_tomo> -r <radius> -s <subsampling> -o <out_dir>')
            print('\t-i (--itomo) <in_tomo> input tomogram (point cloud)')
            print('\t-r (--rad) <radius> radius to connect points in the graph')
            print('\t-s (--subsam) <subsampling> radius of subsampling (optional, default no subsampling)')
            print('\t-o (--odir) <out_dir> putput directory')
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

    if in_tomo is not None:
        print('\t-Loading input tomogram:', in_tomo)
        if os.path.splitext(in_tomo)[1] == '.mrc':
            T = lio.load_mrc(in_tomo)
        else:
            T = nrrd.read(in_tomo)[0]
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
    [coords, graph_array] =make_skeleton_graph(T,r,s)
    print('Spliting in components')
    [graph_ar_comps,coords_comps]=split_into_components(graph_array,coords)
    print('Making graphs line like and save')
    for i in range(len(graph_ar_comps)):
        print('Procesing tubule ',str(i))
        print('Removing cycles')
        L_graph=remove_cycles(graph_ar_comps[i])
        L_graph=remove_cycles(L_graph)
        print('Make polydata')
        Targets, Sources = (L_graph.nonzero())
        points_poly = make_graph_polydata(coords_comps[i],Sources,Targets)
        print('Saving')
        save_vtp(points_poly, out_dir + os.path.splitext(in_tomo)[0]+ '_skel_graph_tubule_'+str(i)+'.vtp')
        print(os.path.splitext(in_tomo)[0]+ '_skel_graph_tubule_'+str(i)+'.vtp'+' saved in '+out_dir)

    print('Successfully terminated. (' + time.strftime("%c") + ')')

if __name__ == "__main__":
    main(sys.argv[1:])

