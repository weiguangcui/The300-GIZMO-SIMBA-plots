import matplotlib
from pylab import *
from readsnapsgl import readsnapsgl
from scipy.spatial import cKDTree
import os
import glob

fpath = "/home2/weiguang/The300/data/catalogues/AHF/GadgetX/"
Xspath = "/home2/weiguang/The300/data/simulation/GadgetX/"

# Cosmologies critical density in 10^10 M_sun/h/(kpc/h)**3
from astropy.cosmology import FlatLambdaCDM
cosm = FlatLambdaCDM(H0=67.77, Om0=0.307115)


snapname = ["snap_128"]
# snapname = []
# for i in np.arange(128,120,-1):
#     exts='000'+str(i)
#     snapname.append('snap_'+exts[-3:])
# #snapname.remove("snap_108");snapname.remove("snap_105");
# print(snapname)


for i, sname in enumerate(snapname):
    print(i,sname)
    
    # load halo catalog has been calculated
    ahfhalo = np.load('/home2/weiguang/Project-300-Clusters/Halo_mass_function_mass-difference/GadgetX/G3X_Mass_'+sname+'info.npy')
    ##ReginIDs[0] HIDs[1] HostID[2] Mvir(3) Xc(4)   Yc(5)   Zc(6)  Rvir(7) fMhires(8) cNFW (9) Mgas200[10] M*200[11] M500[12]  R500[13] fgas500[14] f*500[15]
    ## need to add M*30[16] M*50[17] M*0.1R500[18] projected M*30[19] M*50[20] M*0.1R500[21]
    ahfhalo[:,12]*=1e10  # to Msun/h!!!
    print(ahfhalo[:5,12])
    head = readsnapsgl(Xspath+"NewMDCLUSTER_0001/"+sname, "HEAD",quiet=True)
    if head.Redshift<1.0e-6:
        head.Redshift=0
    Hubblez = head.HubbleParam * np.sqrt(head.OmegaLambda + head.Omega0*(1.+head.Redshift)**3)*100./1000.  #now in Km/s/kpc
    ahfhalo=ahfhalo[(ahfhalo[:,2]<=0)&(ahfhalo[:,3]>1.0e13)]
        
    DX200=np.zeros((ahfhalo.shape[0],22))
    DX200[:,:16] = ahfhalo
    print(DX200.shape)
    
    Rids= np.int32(DX200[:,0]+0.1)
    for n in np.arange(1,325):
        if ((n == 10) and (sname == 'snap_100')) or ((n == 228) and (sname == 'snap_110')):
            continue
        exts='0000'+str(n)
        cn = 'NewMDCLUSTER_'+exts[-4:]
        
        spos = readsnapsgl(Xspath+cn+"/"+sname, "POS ",quiet=True,ptype=4)
        smas = readsnapsgl(Xspath+cn+"/"+sname, "MASS", fullmass=True,quiet=True,ptype=4)
#         svel = readsnapsgl(Xspath+cn+"/"+sname, "VEL ", quiet=True,ptype=4)/np.sqrt(1+head.Redshift)
        smtree = cKDTree(spos)
        
        idsh = np.where(Rids == n)[0]
        for j in idsh:
            cc=DX200[j,4:7]; r200=DX200[j,7]
            idin200 = smtree.query_ball_point(cc, r200, workers=4)
            rr=np.sqrt((spos[idin200,0]-cc[0])**2+(spos[idin200,1]-cc[1])**2)
            
            idlist = smtree.query_ball_point(cc, 30*head.HubbleParam, workers=4)
            if len(idlist)>0:
                smm=np.sum(smas[idlist])
                DX200[j,16]=smm*1e10
                DX200[j,19]=np.sum(smas[idin200][rr<=30*head.HubbleParam])*1e10
                
            idlist = smtree.query_ball_point(cc, 50*head.HubbleParam, workers=4)
            if len(idlist)>0:
                smm=np.sum(smas[idlist])
                DX200[j,17]=smm*1e10
                DX200[j,20]=np.sum(smas[idin200][rr<=50*head.HubbleParam])*1e10
                
            idlist = smtree.query_ball_point(cc, 0.1*DX200[j,13], workers=4)
            if len(idlist)>0:
                smm=np.sum(smas[idlist])
                DX200[j,18]=smm*1e10
                DX200[j,21]=np.sum(smas[idin200][rr<=0.1*DX200[j,13]])*1e10
                
    np.save("./data/G3X_Mbh_"+sname+"-projection", DX200)

#Velocity dispersion !!add hubble flow !! minus halo bulk Velocity has no effects on sigma_v!!
##sigam = std(v)= <(v-<v>)^2> = (given <v> = sum(m_i*v_i)/M) ... prove omit = <v^2> - <v>^2
## here <v^2> = sum(m_i*v_i^2)/M, Must in 3D sigma = np.sqrt(sigma_x^2+sigma_y^2+sigma_z^2)
#         hvel    =svel+scfa*Hubblez*spos/HubbleParam
#         hbv     =np.sum(hvel[:idr200]*np.tile(smas[:idr200],(3,1)).T, axis=0)/m200
#         sgm_200=np.sqrt(np.sum((hvel[:idr200]-hbv)**2*np.tile(smas[:idr200],(3,1)).T)/m200)
