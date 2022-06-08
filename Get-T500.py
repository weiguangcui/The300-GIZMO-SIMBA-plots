import h5py
from pylab import *
import os
from scipy.stats import binned_statistic as bst

k2kev=8.61732814974056E-08
xH = 0.76  # hydrogen mass-fraction
yhelium = (1. - xH) / (4 * xH)
v_unit = 1.0e5       # (e.g. 1.0 km/sec)
prtn = 1.67373522381e-24  # (proton mass in g)
bk = 1.3806488e-16        # (Boltzman constant in CGS)

gzs=np.load("./data/GIZMO_Mbh_snap_128-with_BH-info.npy")
#ReginIDs[0] HIDs[1] Mvir(2) Xc(3)   Yc(4)   Zc(5)  Rvir(6) fMhires(7) cNFW (8)  M500[9]  R500[10] fgas[11] f*[12] M*30[13] M*50[14] M*0.1R500[15]
#ReginIDs[0] HIDs[1] HostID[2] Mvir(3) Xc(4)   Yc(5)   Zc(6)  Rvir(7) fMhires(8) cNFW (9) Mgas200[10] M*200[11] M500[12]  R500[13] fgas500[14] f*500[15] 
## M*30[16] M*50[17] M*0.1R500[18] sigma 30[19], sig50[20], sig0.1[21]
## MaxM_BH 30kpc[22] 50kpc[23] 0.1R500[24]; radius to center BH 30kpc[25] 50kpc[26] 0.1R500[27]

#calculate T500 GIZMO
TG500=np.zeros((gzs.shape[0],3),dtype=np.float32)

Gspath="/home2/weiguang/data7/Gizmo-Simba/"
sname="/snap_128.hdf5"  # redshift 0.1 - 124
for n in np.arange(1,325):
    print(n)
    exts="0000"+str(n)
    cn = "NewMDCLUSTER_"+exts[-4:]
    gsnap=h5py.File(Gspath+cn+sname, 'r')
    sfr = gsnap['PartType0/StarFormationRate'][:]
    Nwl = gsnap['PartType0/NWindLaunches'][:]
    ids = (sfr<0.1) & (Nwl<1)
    tmp = gsnap['PartType0/InternalEnergy'][ids]  #readsnapsgl(Gspath+cn+sname, "TEMP",quiet=True)[ids].astype(np.float64)*k2kev


    NE = gsnap['PartType0/ElectronAbundance'][ids]
#         NE = read_block(npf, "NE  ", endian, 1, longid, fmt, pty, rawdata)

#         we assume it is NR run with full ionized gas n_e/nH = 1 + 2*nHe/nH
#             if mu is None:
#                 mean_mol_weight = (1. + 4. * yhelium) / (1. + 3 * yhelium + 1)
    mean_mol_weight = (1. + 4. * yhelium) / (1. + yhelium + NE)
    tmp = tmp * (5. / 3 - 1) * v_unit**2 * prtn * mean_mol_weight * k2kev / bk

    ids2 = tmp>=0.
    pos = gsnap['PartType0/Coordinates'][:][ids] # readsnapsgl(Gspath+cn+sname, "POS ",quiet=True, ptype=0)[ids]
    mas = gsnap['PartType0/Masses'][ids] # readsnapsgl(Gspath+cn+sname, "MASS", ptype=0,quiet=True)[ids].astype(np.float64)
    rho = gsnap['PartType0/Density'][ids] # readsnapsgl(Gspath+cn+sname, "RHO ",quiet=True)[ids].astype(np.float64)

    idr = np.where(np.int32(gzs[:,0]+0.3)==n)[0]
    for i in idr:
        radii = np.sqrt(np.sum((pos-gzs[i,4:7])**2, axis=1))
        idsr = radii<=gzs[i,13]
        TG500[i,:]=[np.mean(tmp[idsr]), np.sum(tmp[idsr]*mas[idsr])/mas[idsr].sum(),
                    np.sum(rho[idsr&ids2]*mas[idsr&ids2]*tmp[idsr&ids2]**0.25)/np.sum(rho[idsr&ids2]*mas[idsr&ids2]*tmp[idsr&ids2]**-0.75)]
np.save("./data/GIZMO-TM500-fullsample.npy", TG500)