from numpy import ndarray
from supression import nonmaxsup,desyevv
import ctypes
import sys, getopt, time
from graph_uts import *
from mt import lio
from scipy.ndimage import gaussian_filter

def diff3d(T,k):
    """

    Args:
        T:
        k:

    Returns:

    """
    [Nx,Ny,Nz]=np.shape(T)
    Idp = np.zeros((Nx,Ny,Nz))
    Idn = np.zeros((Nx, Ny, Nz))

    if k==0:
        Idp[0:Nx-2, :, :] = T[1:Nx-1, :, :]
        Idn[1:Nx-1, :, :] = T[0:Nx-2, :, :]
        #Pad extremes
        Idp[Nx-1,:,:] = Idp[Nx-2, :, :]
        Idp[0,:,:] = Idp[1,:,:]
    elif k==1:
        Idp[:, 0:Ny - 2, :] = T[:, 1:Ny - 1, :]
        Idn[:, 1:Ny - 1, :] = T[:, 0:Ny - 2, :]
        # Pad extremes
        Idp[:, Ny - 1, :] = Idp[:, Ny - 2, :]
        Idp[:, 0, :] = Idp[:, 1, :]
    else:
        Idp[:, :, 0:Nz - 2] = T[:, :, 1:Nz - 1]
        Idn[:, :, 1:Nz - 1] = T[:, :, 0:Nz - 2]
        # Pad extremes
        Idp[:, :, Nz - 1] = Idp[:, :, Nz - 2]
        Idp[:, :, 0] = Idp[:, :, 1]

    return((Idp-Idn)*0.5)

def eig3dk(Ixx,Iyy,Izz,Ixy,Ixz,Iyz):
    """

    Args:
        Ixx:
        Iyy:
        Izz:
        Ixy:
        Ixz:
        Iyz:

    Returns:

    """
    [Nx,Ny,Nz] = np.shape(Ixx)
    Ixx = Ixx.flatten()

    print(np.shape(Ixx))
    print(Ixx[0])
    Iyy = Iyy.flatten()
    print(np.shape(Iyy))
    Izz = Izz.flatten()
    print(np.shape(Izz))
    Ixy = Ixy.flatten()
    print(np.shape(Ixy))
    Ixz = Ixz.flatten()
    print(np.shape(Ixz))
    Iyz = Iyz.flatten()
    print(np.shape(Iyz))
    result= desyevv(Ixx,Iyy,Izz,Ixy,Ixz,Iyz)
    L1, L2, L3, V1x, V1y, V1z, V2x, V2y, V2z, V3x, V3y, V3z = result
    print(L1)

    L1 = np.reshape(L1, (Nx, Ny, Nz))

    L2 = np.reshape(L2, (Nx, Ny, Nz))
    L3 = np.reshape(L3, (Nx, Ny, Nz))

    V1x = np.reshape(V1x, (Nx, Ny, Nz))
    V1y = np.reshape(V1y, (Nx, Ny, Nz))
    V1z = np.reshape(V1z, (Nx, Ny, Nz))

    V2x = np.reshape(V2x, (Nx, Ny, Nz))
    V2y = np.reshape(V2y, (Nx, Ny, Nz))
    V2z = np.reshape(V2z, (Nx, Ny, Nz))

    V3x = np.reshape(V3x, (Nx, Ny, Nz))
    V3y = np.reshape(V3y, (Nx, Ny, Nz))
    V3z = np.reshape(V3z, (Nx, Ny, Nz))
    print("Resultados guardados")
    return [L1,L2,L3,V1x,V1y,V1z,V2x,V2y,V2z,V3x,V3y,V3z]

def nonmaxsup_fun (I, M, V1x, V1y, V1z, V2x, V2y, V2z):
    """
    
    Args:
        I: 
        M: 
        V1x: 
        V1y: 
        V1z: 
        V2x: 
        V2y: 
        V2z: 

    Returns:

    """
    print('Preparamos mascara')
    [Nx,Ny,Nz]=np.shape(I)
    H= np.zeros((Nx,Ny,Nz))
    H[1:Nx-2,1:Ny-2,1:Nz-2]=1
    M=M*H
    del(H)
    print('Aplanamos vectores')
    Ir = I.flatten()
    V1xr = V1x.flatten()
    V1yr = V1y.flatten()
    V1zr = V1z.flatten()
    V2xr = V2x.flatten()
    V2yr = V2y.flatten()
    V2zr = V2z.flatten()
    Mr=M.flatten().astype('int64')
    dim=np.array([Nx,Ny]).astype('uint32')
    print('Lazamos funcion')
    Br=nonmaxsup(Ir,V1xr,V1yr,V1zr,V2xr,V2yr,V2zr,Mr,dim)
    del (Ir)
    del (V1xr)
    del (V1yr)
    del (V1zr)
    del (V2xr)
    del (V2yr)
    del (V2zr)
    del (Mr)

    B=np.reshape(Br,(Nx,Ny,Nz))
    return(B)

def main(argv):
    # Input parsing
    in_tomo,s=None,None
    eval,evec=None,None
    f=None
    out_tomo=None
    try:
        opts, args = getopt.getopt(argv, "hi:s:e:v:f:o:",["help","itomo","sdesv","eval","evec","filt","otomo"])
    except getopt.GetoptError:
        print('python apply_nonmaxsup.py -i <in_tomo> -s <smooth_desv> -e <eigenvalues> -v <eigenvectors> -f <filter> -o <out_tomo>')
        sys.exit()
    for opt, arg in opts:
        if opt in ("-h","--help"):
            print('python trace_graph.py -i <in_tomo> -r <radius> -s <subsampling> -o <out_dir>')
            print('\t-i (--itomo) <in_tomo> input tomogram')
            print('\t-s (--sdesv) <smooth_desviation> desviation for gaussian filter (~1/3 tubule radium')
            print('\t-e (--eval) <eigenvalues> type of eigenvalues "Hessian" or "Struct" (optional, default "Hessian")')
            print('\t-v (--evec) <eigenvectors> type of eigenvectors "Hessian" or "Struct" (optional, default "Struct")')
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
                print('The input eval must be "Hessian" or "Struct"!')
                sys.exit()
        elif opt in ("-v", "evec"):
            evec = arg
            if not (evec in ('Hessian', 'Struct')):
                print('The input evec must be "Hessian" or "Struct"!')
                sys.exit()
        elif opt in ("-f","--filt"):
            f = arg
        elif opt in ("-o","--otomo"):
            out_tomo = arg
            if not(os.path.splitext(out_tomo)[1] in ('.mrc', '.nhdr', '.nrrd')):
                print('The output file must have a .mrc, .nhdr or nrrd extension!')
                sys.exit()
        else:
            print('python apply_nonmaxsup.py -i <in_tomo> -s <smooth_desv> -e <eigenvalues> -v <eigenvectors> -f <filter> -o <out_tomo>')
            print("Inputs must be one of them!")
            sys.exit()

    if in_tomo is not None:
        print('\t-Loading input tomogram:', in_tomo)
        if os.path.splitext(in_tomo)[1] == '.mrc':
            T = lio.load_mrc(in_tomo)
        else:
            T = nrrd.read(in_tomo)[0]
    else:
        print('python apply_nonmaxsup.py -i <in_tomo> -s <smooth_desv> -e <eigenvalues> -v <eigenvectors> -f <filter> -o <out_tomo>')
        print('Almost an input tomogram -i must be provided!')
        sys.exit()
    if s is not None:
        s=float(s)
        print('Smooth desviation: ',str(s))
    else:
        print('python apply_nonmaxsup.py -i <in_tomo> -s <smooth_desv> -e <eigenvalues> -v <eigenvectors> -f <filter> -o <out_tomo>')
        print('Almost an input sdesv parameter -s must be provided!')
        sys.exit()
    if eval is not None:
        print('Use ',eval, ' eigenvalues')
    else:
        eval="Hessian"
        print('Default: Use ', eval, ' eigenvalues')
    if evec is not None:
        print('Use ',evec, ' eigenvalues')
    else:
        evec="Struct"
        print('Default: Use ', evec, ' eigenvalues')
    if f is not None:
        print ('Trheshold filter in the masc is ',str(f))
    else:
        f=0.5
        print('Default: Trheshold filter in the masc is ', str(f))
    if out_tomo is not None:
        print ('Output tomogram: ',out_tomo)
    else:
        print('python apply_nonmaxsup.py -i <in_tomo> -s <smooth_desv> -e <eigenvalues> -v <eigenvectors> -f <filter> -o <out_tomo>')
        print('Almost an output tomogram -o must be provided!')
        sys.exit()


    #Quick smooth
    print('Making the gaussian smooth')
    T=gaussian_filter(T,s)

    #Construct tensors:
    print('Calculating Tensors')
    Tx = diff3d(T, 0)

    Ty = diff3d(T, 1)
    Tz = diff3d(T, 2)

    print ('Hessian tensor')
    Txx = diff3d(Tx, 0)
    Tyy = diff3d(Ty, 1)
    Tzz = diff3d(Tz, 2)
    Txy = diff3d(Tx, 1)
    Txz = diff3d(Tx, 2)
    Tyz = diff3d(Ty, 2)



    print('Structure tensor')
    Tsxx = Tx*Tx
    Tsyy = Ty*Ty
    Tszz = Tz*Tz
    Tsxy = Tx*Ty
    Tsxz = Tx*Tz
    Tsyz = Ty*Tz



    print('Calculating eigenvalues')
    [L1, L2, L3, V1x, V1y, V1z, V2x, V2y, V2z, V3x, V3y, V3z]=eig3dk(Txx,Tyy,Tzz,Txy,Txz,Tyz)
    [Ls1, Ls2, Ls3, Vs1x, Vs1y, Vs1z, Vs2x, Vs2y, Vs2z, Vs3x, Vs3y, Vs3z] = eig3dk(Tsxx, Tsyy, Tszz, Tsxy, Tsxz, Tsyz)
    lio.write_mrc(L1.astype(np.float32), '/project/chiem/pelayo/neural_network/Ctrl_20220511_368d_tomo06_reg_synth_L1.mrc')


    del L2
    del L3
    del Ls1
    del Ls2
    del Ls3
    del V1x
    del V1y
    del V1z
    del V2x
    del V2y
    del V2z
    del V3x
    del V3y
    del V3z
    del Vs3x
    del Vs3y
    del Vs3z
    print(np.max(L1))
    L1=-L1
    print('Calcula M?')
    M=np.where(L1>f,1,0)
    print(M)


    print('Applying the nonmaximal suppresion' )
    P = nonmaxsup_fun(L1,M,Vs1x,Vs1y,Vs1z,Vs2x,Vs2y,Vs2z)

    print('Saving')
    if os.path.splitext(out_tomo)[1] == '.mrc':
        lio.write_mrc(P,out_tomo)
    else:
        nrrd.write(out_tomo,P)

    print('Successfully terminated. (' + time.strftime("%c") + ')')


if __name__ == "__main__":
    main(sys.argv[1:])