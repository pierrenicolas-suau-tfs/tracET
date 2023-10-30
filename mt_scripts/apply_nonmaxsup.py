
from supression import nonmaxsup,desyevv

import sys, getopt, time
from graph_uts import *
from core import lio


def angauss(I,s,r=1):
    """

    :param I:
    :param s:
    :param r:
    :return:
    """
    #Initialitation
    [Nx, Ny, Nz] = np.shape(I)
    if np.remainder(Nx,2)==1:
        Nx2=np.floor(0.5*Nx)
        Vnx=np.arange(-Nx2,Nx2+1)
    else:
        Nx2=0.5*Nx
        Vnx=-np.arange(-Nx2,Nx2)
    if np.remainder(Ny,2)==1:
        Ny2=np.floor(0.5*Ny)
        Vny=np.arange(-Ny2,Ny2+1)
    else:
        Ny2=0.5*Ny
        Vny=-np.arange(-Ny2,Ny2)
    if np.remainder(Nz,2)==1:
        Nz2=np.floor(0.5*Nz)
        Vnz=np.arange(-Nz2,Nz2+1)
    else:
        Nz2=0.5*Nz
        Vnz=-np.arange(-Nz2,Nz2)

    [X,Y,Z] = np.meshgrid(Vny,Vnx,Vnz)
    A = 1./((s*s*np.sqrt(r*(2.*np.pi)**3.)))
    a = 1./(2.*s*s)
    b=a/r

    #Kernel
    K=A*np.exp(-a*(X*X+Y*Y)-b*Z*Z)

    #Convolution
    F=np.fft.fftshift(np.fft.ifftn(np.fft.fftn(I)*np.fft.fftn(K)))

    return F

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
    Ixx = np.swapaxes(Ixx,0,2).flatten()


    Iyy = np.swapaxes(Iyy,0,2).flatten()

    Izz = np.swapaxes(Izz,0,2).flatten()

    Ixy = np.swapaxes(Ixy,0,2).flatten()

    Ixz = np.swapaxes(Ixz,0,2).flatten()

    Iyz = np.swapaxes(Iyz,0,2).flatten()

    result= desyevv(Ixx,Iyy,Izz,Ixy,Ixz,Iyz)
    L1, L2, L3, V1x, V1y, V1z, V2x, V2y, V2z, V3x, V3y, V3z = result


    L1 = np.swapaxes(np.reshape(L1, (Nz, Ny, Nx)),0,2)

    L2 = np.swapaxes(np.reshape(L2, (Nz, Ny, Nx)),0,2)
    L3 = np.swapaxes(np.reshape(L3, (Nz, Ny, Nx)),0,2)

    V1x = np.swapaxes(np.reshape(V1x, (Nz, Ny, Nx)),0,2)
    V1y = np.swapaxes(np.reshape(V1y, (Nz, Ny, Nx)),0,2)
    V1z = np.swapaxes(np.reshape(V1z, (Nz, Ny, Nx)),0,2)

    V2x = np.swapaxes(np.reshape(V2x, (Nz, Ny, Nx)), 0, 2)
    V2y = np.swapaxes(np.reshape(V2y, (Nz, Ny, Nx)), 0, 2)
    V2z = np.swapaxes(np.reshape(V2z, (Nz, Ny, Nx)), 0, 2)

    V3x = np.swapaxes(np.reshape(V3x, (Nz, Ny, Nx)), 0, 2)
    V3y = np.swapaxes(np.reshape(V3y, (Nz, Ny, Nx)), 0, 2)
    V3z = np.swapaxes(np.reshape(V3z, (Nz, Ny, Nx)), 0, 2)

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

    [Nx,Ny,Nz]=np.shape(I)
    H= np.zeros((Nx,Ny,Nz))
    H[1:Nx-2,1:Ny-2,1:Nz-2]=1
    M=M*H
    del H
    M = np.swapaxes(M,0,2).flatten().astype(int)
    Mr=np.arange(0,Nx*Ny*Nz,dtype=int)
    Mr=Mr[M==1]


    Ir = np.swapaxes(I,0,2).flatten()
    V1xr = np.swapaxes(V1x,0,2).flatten()
    V1yr = np.swapaxes(V1y,0,2).flatten()
    V1zr = np.swapaxes(V1z,0,2).flatten()
    V2xr = np.swapaxes(V2x,0,2).flatten()
    V2yr = np.swapaxes(V2y,0,2).flatten()
    V2zr = np.swapaxes(V2z,0,2).flatten()

    dim=np.array([Nx,Ny]).astype('uint32')

    Br=nonmaxsup(Ir,V1xr,V1yr,V1zr,V2xr,V2yr,V2zr,Mr,dim)
    del Ir
    del V1xr
    del V1yr
    del V1zr
    del V2xr
    del V2yr
    del V2zr
    del Mr

    B=np.swapaxes(np.reshape(Br,(Nz,Ny,Nx)),0,2)
    H=np.zeros((Nx,Ny,Nz))
    H[2:Nx-2,2:Ny-2,2:Nz-2]=1
    B=B*H
    del H
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
    T=angauss(T,s)

    lio.write_mrc(T.astype(np.float32),
                  '/project/chiem/pelayo/neural_network/compara/python/Ctrl_20220511_368d_tomo06_reg_synth_gauss_python.mrc')

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
    [Lh1, Lh2, Lh3, Vh1x, Vh1y, Vh1z, Vh2x, Vh2y, Vh2z, Vh3x, Vh3y, Vh3z]=eig3dk(Txx,Tyy,Tzz,Txy,Txz,Tyz)
    [Ls1, Ls2, Ls3, Vs1x, Vs1y, Vs1z, Vs2x, Vs2y, Vs2z, Vs3x, Vs3y, Vs3z] = eig3dk(Tsxx, Tsyy, Tszz, Tsxy, Tsxz, Tsyz)
    if eval == 'Hessian':
        L1 = -Lh1
    else:
        L1 = -Ls1
    if evec == 'Hessian':
        V1x = Vh1x
        V1y = Vh1y
        V1z = Vh1z
        V2x = Vh2x
        V2y = Vh2y
        V2z = Vh2z
    else:
        V1x = Vs1x
        V1y = Vs1y
        V1z = Vs1z
        V2x = Vs2x
        V2y = Vs2y
        V2z = Vs2z

    del Lh1
    del Lh2
    del Lh3
    del Ls1
    del Ls2
    del Ls3
    del Vh1x
    del Vh1y
    del Vh1z
    del Vh2x
    del Vh2y
    del Vh2z
    del Vh3x
    del Vh3y
    del Vh3z
    del Vs1x
    del Vs1y
    del Vs1z
    del Vs2x
    del Vs2y
    del Vs2z
    del Vs3x
    del Vs3y
    del Vs3z


    lio.write_mrc(L1.astype(np.float32),
                  '/project/chiem/pelayo/neural_network/compara/python/Ctrl_20220511_368d_tomo06_reg_synth_L1_python.mrc')
    lio.write_mrc(V1x.astype(np.float32),
                  '/project/chiem/pelayo/neural_network/compara/python/Ctrl_20220511_368d_tomo06_reg_synth_V1x_python.mrc')
    lio.write_mrc(V1y.astype(np.float32),
                  '/project/chiem/pelayo/neural_network/compara/python/Ctrl_20220511_368d_tomo06_reg_synth_V1y_python.mrc')
    lio.write_mrc(V1z.astype(np.float32),
                  '/project/chiem/pelayo/neural_network/compara/python/Ctrl_20220511_368d_tomo06_reg_synth_V1z_python.mrc')
    lio.write_mrc(V2x.astype(np.float32),
                  '/project/chiem/pelayo/neural_network/compara/python/Ctrl_20220511_368d_tomo06_reg_synth_V2x_python.mrc')
    lio.write_mrc(V2y.astype(np.float32),
                  '/project/chiem/pelayo/neural_network/compara/python/Ctrl_20220511_368d_tomo06_reg_synth_V2y_python.mrc')
    lio.write_mrc(V2z.astype(np.float32),
                  '/project/chiem/pelayo/neural_network/compara/python/Ctrl_20220511_368d_tomo06_reg_synth_V2z_python.mrc')


    M=np.zeros_like(L1)
    M[L1>f] = 1
    lio.write_mrc(M.astype(np.float32), '/project/chiem/pelayo/neural_network/compara/python/Ctrl_20220511_368d_tomo06_reg_synth_M_python.mrc')


    print('Applying the nonmaximal suppresion' )
    P = nonmaxsup_fun(L1,M,V1x,V1y,V1z,V2x,V2y,V2z)

    print('Saving')
    if os.path.splitext(out_tomo)[1] == '.mrc':
        lio.write_mrc(P.astype(np.float32), out_tomo)
    else:
        nrrd.write(out_tomo,P)

    print('Successfully terminated. (' + time.strftime("%c") + ')')


if __name__ == "__main__":
    main(sys.argv[1:])