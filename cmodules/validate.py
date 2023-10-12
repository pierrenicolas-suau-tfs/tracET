from mt import lio
import numpy as np

T_py=lio.load_mrc('/data/chiem/pelayo/neural_network/compara/python/Ctrl_20220511_368d_tomo06_reg_synth_supred_python.mrc')
T_mat=lio.load_mrc('/data/chiem/pelayo/neural_network/compara/matlab/Ctrl_20220511_368d_tomo06_reg_synth_supred_matlab.mrc')
compare=0
python=0
matlab=0
[Nx,Ny,Nz]=np.shape(T_py)
for x in range(Nx):
    for y in range(Ny):
        for z in range(Nz):
            if T_py[x,y,z]!=T_mat[x,y,z]:
                print('En ['+str(x)+','+str(y)+','+str(z)+'] es distinto!')
                print('python vale '+str(T_py[x,y,z]))
                print('matlab vale '+str(T_mat[x,y,z]))
                compare=compare+1
                if T_py[x,y,z]==1:
                    python+=1
                else:
                    matlab+=1
print('diferencias '+str(compare))
print('difs in python '+str(python))
print('difs in matlab '+str(matlab))
difs_rare=python-matlab
per_errot_tot=100*difs_rare/(Nx*Ny*Nz)
per_error_pos=100*difs_rare/np.sum(T_py)
print('Error desconocido de '+str(per_errot_tot)+' %')
print('Error sobre positivos de '+str(per_error_pos)+' %')