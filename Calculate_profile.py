# import matplotlib
# from pylab import *
import numpy as np
import h5py
from readsnapsgl import readsnapsgl
from scipy.spatial import cKDTree
import os, sys
import glob

# fpath = "/home2/weiguang/MUSIC/MUSIC_PLANCK/" # for orignial data
fpath = "/home2/weiguang/The300/data/catalogues/AHF/GIZMO/"
Xspath = "/home2/weiguang/data7/Gizmo-Simba/"

mean_mol_weight= 0.588
prtn = 1.67373522381e-24  # (proton mass in g)
bk = 1.3806488e-16        # (Boltzman constant in CGS)
progenIDs=np.loadtxt("../Halo_mass_function_mass-difference/GIZMO/Progenitor-IDs-for-center-cluster.txt",dtype=np.int64)
sn_sa_red=np.loadtxt('../redshifts.txt')

# Cosmologies critical density in 10^10 M_sun/h/(kpc/h)**3
from astropy.cosmology import FlatLambdaCDM
cosm = FlatLambdaCDM(H0=67.77, Om0=0.307115)
# rhoc500=500*cosm.critical_density0.to("M_sun/kpc**3")/1.0e10/0.6777**2

##Open h5 file for write
h5fname="data/CGM-profiles-progenitors.hdf5"
if not os.path.isfile(h5fname):
    fh5w = h5py.File(h5fname, "w")
    fh5w.close()
    
# if len(sys.argv) != 3:
#     raise ValueError('Usage: python3 Get-M200-M500-AHF.py cluster_region_num snapshot_number')
# else:
#     crn = int(sys.argv[1])
#     cn = 'NewMDCLUSTER_%04d'%crn
#     sn = int(sys.argv[2])
#     snapn='snap_%03d'%sn

sn = 128
snapn='snap_%03d'%sn

red=sn_sa_red[sn,2]
rhoc500=500*cosm.critical_density(red).to("M_sun/kpc**3")/1.0e10/0.67778**2*(1./(1+red))**3 #to comoving

for crn in np.arange(1,325):
    cn = 'NewMDCLUSTER_%04d'%crn

    #h5 write files
    fh5w = h5py.File(h5fname, "r+")
    if cn in fh5w.keys():
        cgrp = fh5w[cn]
    else:
        cgrp = fh5w.create_group(cn)
    if snapn in cgrp.keys():
        sngrp=cgrp[snapn]
    else:
        sngrp = cgrp.create_group(snapn)
    
    #Now we use the cluster withID
    if (progenIDs[crn-1, sn]<1):
        print('No progenitors at ',cn, snapn)
        continue
    else:
        tmpd=np.load('../Halo_mass_function_mass-difference/GIZMO/GS_Mass_snap_'+str(sn)+'info.npy')
        #ReginIDs HIDs  HosthaloID Mvir(4) Xc(6)   Yc(7)   Zc(8)  Rvir(12) fMhires(38) cNFW (42) Mgas200 M*200 M500  R500 fgas500 f*500
        ids = np.where((np.int32(tmpd[:,0])==crn) & (np.int64(tmpd[:,1]) == progenIDs[crn-1, sn]))[0]
        if len(ids) != 1:
            print("can not find central galaxy for ", crn, snapn, "len(ids):", len(ids))
            continue

    cc = tmpd[ids, 4:7][0]
    r200 = tmpd[ids, 7][0]
    r500 = tmpd[ids, 13][0]
    M500 = tmpd[ids, 12][0]
    M200 = tmpd[ids,3][0]
    print(M200, r200, cc)
#     head="M200: {:.3e} [Msun/h], R200 {:.3f} [kpc/h], M500: {:.3e} [Msun/h], R500 {:.3f} [kpc/h], center: {:.3f} {:.3f} {:.3f} [kpc/h] 250 ratial bins from 0.001 - 1.5 R200 \n gas density [Msun/h/(kpc/h)^3]; MW temp [k]; Pressure []; electron number density []; entropy []; metal [];".format( M200, r200, cc[0], cc[1], cc[2] )

    #h5 write files
    sngrp.attrs['center_x'] = cc[0]; sngrp.attrs['center_y'] = cc[1]; sngrp.attrs['center_x'] = cc[2];
    sngrp.attrs['r200'] = r200; sngrp.attrs['r500'] = r500;
    sngrp.attrs['M200]'] = M200; sngrp.attrs['M500]'] = M500;
    sngrp.attrs['README'] = "all distance units are in kpc/h, all mass units are in M_sun/h.\n The radial bins are given by rbins=np.logspace(np.log10(0.001*r200), np.log10(1.5*r200), num=251)\n Profile units: Gdens -- gas density [M_sun/h/(kpc/h)^3]; MWTemp -- mass weighted temperature k; Pressure -- gas pressure [Msun/h * k/ (kpc/h)^3]"
    
    #simulation
    f=h5py.File(Xspath+cn+'/'+snapn+'.hdf5', 'r')

    v_unit = 1.0e5 * np.sqrt(f['Header'].attrs['Time'])       # (e.g. 1.0 km/sec)
    # tpos=np.empty((0,3), dtype=np.float32)
    # tmas=np.empty(0, dtype=np.float32)
    # for j in range(6):
    #     tpos = np.append(tpos, f['PartType'+str(j)+'/Coordinates'][:], axis=0)
    #     tmas = np.append(tmas, f['PartType'+str(j)+'/Masses'][:])
    # trr = np.sqrt(np.sum((tpos-cc)**2, axis=1))
    # idt=trr<3000
    # trr=trr[idt]
    # tmas=tmas[idt]*1.0e10

    j = 0
    pos = f['PartType'+str(j)+'/Coordinates'][:]
    rr = np.sqrt(np.sum((pos-cc)**2, axis=1))
    mas = f['PartType'+str(j)+'/Masses'][:]
    sgden=f['PartType'+str(j)+'/Density'][:] * 1.0e10 /0.6777/(1/0.6777)**3  #in Msun/kpc^3
    temp= f['PartType'+str(j)+'/InternalEnergy'][:] * (5. / 3 - 1) * v_unit**2 * prtn * mean_mol_weight / bk
    wind= f['PartType'+str(j)+'/NWindLaunches'][:]
    sfr = f['PartType'+str(j)+'/StarFormationRate'][:]
    metl= f['PartType'+str(j)+'/Metallicity'][:]
    metl = metl[:,0]

    ids = np.where((temp>1.0e6)&(rr<1.5*r200)&(wind<1)&(sgden<2.88e6))[0]
    print(len(ids))
    f.close()

    rr=rr[ids]
    mas=mas[ids]*1.0e10
    temp=temp[ids]

    rbins=np.logspace(np.log10(0.001*r200), np.log10(1.5*r200), num=251)
    vol=4*np.pi*rbins**3/3
    vol=vol[1:]-vol[:-1]
    dens,xe=np.histogram(rr, bins=rbins, weights=mas)
    tem,xe =np.histogram(rr, bins=rbins, weights=mas*temp)
    # tden,xe=np.histogram(trr, bins=rad, weights=tmas)

    
    #h5 write files:  profiles that need to be calculated, gas density; MW temperature; Pressure; metallicity; entropy; electron number density; stellar density; total density; 
    # data=np.zeros((dens.size,4),dtype=np.float64)
    # data[:,0] = (rad[1:]+rad[:-1])/2
    if 'Gdens' in sngrp.keys():
        sngrp['Gdens']=dens/vol
    else:
        sngrp.create_dataset('Gdens', data = dens/vol)   # density in Msun/h/kpc/h^3
    if 'MWTemp' in sngrp.keys():
        sngrp['MWTemp']=tem/dens
    else:
        sngrp.create_dataset('MWTemp', data = tem/dens)   # MW temp
    if 'Pressure' in sngrp.keys():
        sngrp['Pressure']=tem/vol
    else:
        sngrp.create_dataset('Pressure', data = tem/vol)   # pressure
    
    #h5files
    fh5w.close()

# np.savetxt("data/GIZMO_profiles/"+cn+"_"+snapn+".txt", data, header=head) 
