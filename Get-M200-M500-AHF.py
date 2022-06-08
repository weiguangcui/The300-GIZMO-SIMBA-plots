##THis somehow give wrong Halos informations!!! please use the Get-M500-BH-GIZMO.py!!!

import matplotlib
from pylab import *
import h5py
from readsnapsgl import readsnapsgl
from scipy.spatial import cKDTree
import os, sys
import glob

# fpath = "/home2/weiguang/MUSIC/MUSIC_PLANCK/" # for orignial data
fpath = "/home2/weiguang/The300/data/catalogues/AHF/GIZMO/"
Xspath = "/home2/weiguang/data7/Gizmo-Simba/"

# Cosmologies critical density in 10^10 M_sun/h/(kpc/h)**3
from astropy.cosmology import FlatLambdaCDM
cosm = FlatLambdaCDM(H0=67.77, Om0=0.307115)
# rhoc500=500*cosm.critical_density0.to("M_sun/kpc**3")/1.0e10/0.6777**2
rhoc500_z00=500*cosm.critical_density0.to("M_sun/kpc**3")/1.0e10/0.6777**2
rhoc500_z05=500*cosm.critical_density(0.523).to("M_sun/kpc**3")/1.0e10/0.6777**2*(1./1.523)**3 #to comoving
rhoc500_z10=500*cosm.critical_density(1.031).to("M_sun/kpc**3")/1.0e10/0.6777**2*(1./2.031)**3
rhoc500_z23=500*cosm.critical_density(2.302).to("M_sun/kpc**3")/1.0e10/0.6777**2*(1./3.302)**3
rhoc500_z40=500*cosm.critical_density(4.018).to("M_sun/kpc**3")/1.0e10/0.6777**2*(1./5.018)**3

sn_sa_red=np.loadtxt('../redshifts.txt')

if len(sys.argv) != 2:
    raise ValueError('Usage: python3 Get-M200-M500-AHF.py snapshot_number')
else:
    sn = int(sys.argv[1])
    snapn='snap_%03d'%sn

red=sn_sa_red[sn,2]
rhoc500=500*cosm.critical_density(red).to("M_sun/kpc**3")/1.0e10/0.67778**2*(1./(1+red))**3 #to comoving

# for i, sname in enumerate(["/snap_128.hdf5", "/snap_109.hdf5", "/snap_096.hdf5", "/snap_074.hdf5", "/snap_055.hdf5"]):  # redshift 0.1 - 124 "/snap_128", "/snap_109", 
DX25=np.zeros((900000,16),dtype=np.float64)
#ReginIDs[0] HIDs[1] Mvir(2) Xc(3)   Yc(4)   Zc(5)  Rvir(6) fMhires(7) cNFW (8)  M500[9]  R500[10] fgas[11] f*[12] M*30[13] M*50[14] M*0.1R500[15]
ct=0
for n in np.arange(1,325):
    print(n)
    exts='0000'+str(n)
    cn = 'NewMDCLUSTER_'+exts[-4:]

    f = glob.glob(fpath + cn + "/GIZMO-NewMDCLUSTER_" +cn[-4:]+"."+snapn+".*.AHF_halos")
    if (len(f) == 0):
        raise ValueError('Can not find AHF halo catalogue for '+cn+reds[i])
    f=f[0]
    tmpd = np.loadtxt(f, usecols=(0,1,3,5,6,7,11,37,42),dtype=np.float64)
    ids = np.where((tmpd[:,1]==0)&(tmpd[:,2]>1.0e13)&(tmpd[:,-2]>0.98))[0]
    tmpd = tmpd[ids]
    tmpd[:,1]=tmpd[:,0]
    tmpd[:,0]=n
    DX25[ct:ct+ids.size,:9]=tmpd
    
    DX500=np.copy(tmpd[:,:7]) # M500 R500 fgas f* M*30 M*50 M*0.1R500
    #simulation
    f=h5py.File(Xspath+cn+'/'+snapn+'.hdf5', 'r')

    pos=np.empty((0,3), dtype=np.float32)
    mas=np.empty(0, dtype=np.float32)
    idtp=np.empty(0, dtype=np.int32)
    for j in range(6):
        pos = np.append(pos, f['PartType'+str(j)+'/Coordinates'][:], axis=0)
        tms = f['PartType'+str(j)+'/Masses'][:]
        mas = np.append(mas, tms)
        idtp = np.append(idtp, np.ones(tms.size, dtype=np.int32)*j)
    f.close()
        
    for i in np.arange(ids.size):
        cc = tmpd[i,3:6]
        rr = tmpd[i,6]

        idcs = (pos[:,0]>cc[0]-rr) & (pos[:,0]<cc[0]+rr) &\
                (pos[:,1]>cc[1]-rr) & (pos[:,1]<cc[1]+rr) &\
                (pos[:,2]>cc[2]-rr) & (pos[:,2]<cc[2]+rr)
        tpos = np.copy(pos[idcs])
        tmas = np.copy(mas[idcs])
        tidp = np.copy(idtp[idcs])

        radii = np.sqrt(np.sum((tpos-cc)**2, axis=1))
        idst = np.argsort(radii)
        radii = radii[idst]
        tmas = tmas[idst]
        tidp = tidp[idst]

        dens = np.cumsum(tmas)*3./4./np.pi/radii**3
        id500 = np.where(dens>rhoc500.value)[0]
        if len(id500)<=0:
            mtree = cKDTree(tpos)
            Neib,ii = mtree.query(tpos,k=10)
            cc=tpos[Neib[:,-1].argmin()]-0.001

            radii = np.sqrt(np.sum((tpos-cc)**2, axis=1))
            idst = np.argsort(radii)
            radii = radii[idst]
            tmas = tmas[idst]
            tidp = tidp[idst]

            dens = np.cumsum(tmas)*3./4./np.pi/radii**3
            id500 = np.where(dens>rhoc500.value)[0]            

        r500 = radii[id500[-1]]
        m500 = np.sum(tmas[:id500[-1]])
        mg500 = np.sum(tmas[:id500[-1]][tidp[:id500[-1]]==0])
        ms500 = np.sum(tmas[:id500[-1]][tidp[:id500[-1]]==4])
        ms30 = np.sum(tmas[(radii<30*0.67778) & (tidp==4)])
        ms50 = np.sum(tmas[(radii<50*0.67778) & (tidp==4)])
        mr1 = np.sum(tmas[(radii<r500*0.1) & (tidp==4)])
        DX500[i,:]=np.asarray([m500*1.0e10, r500, mg500/m500, ms500/m500, ms30*1.0e10, ms50*1.0e10, mr1*1.0e10])
    DX25[ct:ct+ids.size,9:] = DX500
    ct+=ids.size
print(ct)
np.save("data/GIZMO_Mass_"+snapn+"_info-AHF", DX25[:ct]) 
