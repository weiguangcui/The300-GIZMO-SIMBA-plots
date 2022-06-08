import caesar
import h5py, os, sys
import yt, pickle
import numpy as np

Gizmo_BCGMevo=np.zeros((324,129),dtype=np.float32)-1
Gizmo_BCGMevo[:,0]=np.arange(1,325) 
mcahf=np.loadtxt('GIZMO-Matched-AHFhalo-Caesar-galaxy.txt') # region ID, AHF ID, Caesar Hid, Gid, dist hpos, gpos, select Mgal in log10
idini=np.int32(mcahf[:,3]+0.001)
for i in np.arange(0,324):
    cln='NewMDCLUSTER_%04d/'%(i+1)
    for j in np.arange(128,29,-1):
        if idini[i]>=0:
            ds=caesar.load('/home2/weiguang/data6/CL300_Caesar/'+cln+'Caesar_snap_%03d.hdf5'%j)
            Gizmo_BCGMevo[i,j]=ds.galaxies[idini[i]].masses['stellar']
            if 'progen_galaxy_star' in ds.galaxies[idini[i]].__dir__():
                if len(ds.galaxies[idini[i]].progen_galaxy_star)>0:
                    idini[i]=ds.galaxies[idini[i]].progen_galaxy_star.min()
                else:
                    idini[i]=-1
            else:
                idini[i]=-1
np.save('GIZMO_BCG_Mg_progen_min', Gizmo_BCGMevo)