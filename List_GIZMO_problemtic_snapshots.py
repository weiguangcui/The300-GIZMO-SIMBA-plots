import h5py
import numpy as np

path='/home2/weiguang/data7/Gizmo-Simba/'

for i in np.arange(1,325):
    sf='NewMDCLUSTER_%04d' % i
    for j in np.arange(0,129):
        sn='snap_%03d.hdf5' % j
        f=h5py.File(path+sf+'/'+sn,'r')
        if (f['PartType0/Coordinates'][:].max()<=0.0):
            print('Coordinate: ', i, j)
        if (f['PartType0/Masses'][:].max()<=0.0):
            print('Mass: ', i, j)
        f.close()
