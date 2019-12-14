#!/usr/bin/env python
from __future__ import absolute_import, division, print_function
import matplotlib
matplotlib.use('Agg')

import numpy as np
import pylab as pl

from srcfinder_util import *
import time

gettime = time.time

def rdn2rgb(rdnrgb,mask,p=0.99):
    rgbpix = rdnrgb[~mask]

    for bi in range(rgbpix.shape[-1]):
        bmin,bmax = extrema(rgbpix[:,bi],p=p)
        if bi!=0:
            rgbmin,rgbmax = max(bmin,rgbmin),min(bmax,rgbmax)
        else:
            rgbmin,rgbmax = bmin,bmax
    return rgbmin,rgbmax
            
def salience2detections(salimg,cmfimg,salthr,cmfthr,cmflid,cmfmap,outdir):
    from pandas import DataFrame
    from scipy.ndimage.measurements import center_of_mass as _cmass

    assert((cmfimg.ndim == 3) and (cmfimg.shape[2]==4))
    
    outhdr  = ['detid','lid','detbbminr','detbbmaxr','detbbminc','detbbmaxc']
    outhdr += ['salmax','salmin','salmed','salmad','salmaxrow','salmaxcol']
    outhdr += ['salmaxlat','salmaxlon']
    outhdr += ['cmfmax','cmfmin','cmfmed','cmfmad','cmfmaxrow','cmfmaxcol']
    outhdr += ['cmfmaxlat','cmfmaxlon']

    print('salmm.shape: "%s"'%str((salmm.shape)))
    salpos = salimg[...,-1]
    if salimg.shape[-1]==2:
        salpos = salpos / salimg.sum(axis=2)
        print('extrema(salimg.sum(axis=2)): "%s"'%str((extrema(salimg.sum(axis=2)))))
        
    print('cmfimg.shape: "%s"'%str((cmfimg.shape)))
    cmfrgb = cmfimg[...,[0,1,2]]
    cmfdet = cmfimg[...,3]
    nodata = cmfrgb[...,0]==-9999
    rgbmin,rgbmax = rdn2rgb(cmfrgb,nodata)

    print('rgbmin,rgbmax: "%s"'%str((rgbmin,rgbmax)))
    cmfmask = cmfdet>cmfthr
    #thalf = 128//4
    #salpos = imrescale(salpos,max(salpos.shape)/float(thalf),order=ORDER_LINEAR)
    print('salpos.shape: "%s"'%str((salpos.shape)))
    print('extrema(salpos): "%s"'%str((extrema(salpos))))
        
        
    print('salimg.shape: "%s"'%str((salimg.shape)))
    salmask = salpos>salthr
    salreg = imlabel(salmask)
    print('salreg: "%s"'%str((salreg)))
    salids = np.unique(salreg[:])[1:]
    print('len(salids): "%s"'%str((len(salids))))
    # skip salids[0] == 0
    df = []
    stime = gettime()
    salobj = findobj(salreg)

    if not pathexists(outdir):
        mkdir(outdir)

    # marker size/color
    msz = 50
    mec,mfc = (0.75,0.75,0.75,0.75),(0.5,0.5,0.5,0.5)
    mapstr = lambda a: map(str,a)
        
    for ri,robj in enumerate(salobj):
        plab = ri+1
        imin,imax = robj[0].start,robj[0].stop
        jmin,jmax = robj[1].start,robj[1].stop
        # compute salience stats with respect to salience mask
        ndmask = ~nodata[robj]
        pmsk = (salreg[robj]==plab) & ndmask
        pimg = salpos[robj].copy()
        pimgm = pimg*pmsk
        ppix = pimg[pmsk]
        pmed = np.median(ppix)
        pmad = mad(ppix,medval=pmed)
        psol = pmsk.sum()/len(ppix)
        ppmn,ppmx = extrema(ppix)
        pmi,pmj = np.int32(_cmass(pimgm==ppmx))+[imin,jmin]

        #salperim = np.float32(findboundaries(pmsk,mode='inner'))
        #salperim[salperim==0] = np.nan
        
        # compute CMF stats with respect to salience and CMF mask
        cmsk = cmfmask[robj] & pmsk
        cimg = cmfdet[robj].copy()
        cimgm = cimg*cmsk
        cpix = cimg[cmsk]
        cpmn,cpmx = extrema(cpix)
        cmed = np.median(cpix)
        cmad = mad(cpix,medval=cmed)
        cmi,cmj = np.int32(_cmass(cimgm==cpmx))+[imin,jmin]
        csol = cmsk.sum()/len(cpix)

        rgbimg = np.clip((cmfrgb[robj]-rgbmin)/(rgbmax-rgbmin),0,1)
        
        # georeference center of mass
        plli,pllj = [p[0] for p in sl2latlon(pmj,pmi,mapinfo=cmfmap)]
        clli,cllj = [p[0] for p in sl2latlon(cmj,cmi,mapinfo=cmfmap)]

        detid = cmflid+'-%d'%plab
        riout = [detid,cmflid,imin,jmin,imax,jmax,
                 ppmx,ppmn,pmed,pmad,pmi,pmj,plli,pllj,
                 cpmx,cpmn,cmed,cmad,cmi,cmj,clli,cllj]
        df.append(riout)
        
        #print(zip(outhdr,riout),(psol,csol))
        smin = max(salthr,ppmn)
        fig,ax = pl.subplots(1,3,sharex=True,sharey=True,figsize=(9,3.5))
        pimg[pimg<salthr] = np.nan
        cimg[cimg<cmfthr] = np.nan
        ax[0].imshow(pimg,vmin=salthr,vmax=1.0,cmap='YlOrRd')
        ax[0].set_title('Salience $\in$ [%.1f,%.1f]%%'%(100*smin,100*ppmx))

        ax[1].imshow(cimg,vmin=cmfthr,vmax=1500,cmap='YlOrRd')
        ax[1].set_title('CMF $\in$ [250,1500] ppmm')        
        
        ax[2].imshow(rgbimg)
        ax[2].set_title('RGBQL')
        
        for axi in ax:
            axi.scatter([cmj-jmin],[cmi-imin],msz,edgecolor=mec,facecolor=mfc)
            xtl = mapstr(axi.get_xticks())
            ytl = mapstr(axi.get_yticks())
            axi.set_xticklabels(shift_ticklabels(xtl,int(jmin)))
            axi.set_yticklabels(shift_ticklabels(ytl,int(imin)))
            axi.set_xlabel('sample index')
            #axi.imshow(salperim,alpha=0.8)

        ax[0].set_ylabel('line index')
        detfigf = pathjoin(outdir,detid+'.pdf')
        pl.savefig(detfigf)
        print('saved',detfigf)

    df = DataFrame.from_records(df,columns=outhdr)
    print('Elapsed time: %.4fs'%(gettime()-stime))
    print(df)

    return df
    
def save_detections(outf,df,sheet='Plume_List'):
    import pandas as pd
    dfcols = \
    ['detid',
     'lid',
     'cmfmaxlat',
     'cmfmaxlon',
     'cmfmin',
     'cmfmax',
     'cmfmed',
     'cmfmad',
     'salmin',
     'salmax',
     'salmed',
     'salmad',
    ]

    outcols = \
    ['Candidate ID',
     'Line name',
     'Plume Latitude (deg)',
     'Plume Longitude (deg)',
     'CMF Min (ppmm)',
     'CMF Max (ppmm)',
     'CMF Median (ppmm)',
     'CMF MAD (ppmm)',
     'Salience Min (%)',
     'Salience Max (%)',
     'Salience Median (%)',
     'Salience MAD (%)'
    ]

    dfout = pd.DataFrame.from_records(df.loc[:,dfcols].values,columns=outcols,
                                      index=outcols[0])
    
    writer = pd.ExcelWriter(outf)
    dfout.to_excel(writer,sheet)
    writer.save()

    # also save a csv copy
    dfout.to_csv(splitext(outf)[0]+'.csv')
                  
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Salience Map -> Prediction Summary")

    
    model_version = 'v2'
    probthr = 0.5
    cmfthr = 250
    parser.add_argument('--prob_thr', type=float,  default=probthr,
                        help='Salience threshold (default=%.2f)'%probthr)
    parser.add_argument('--ppmm_thr', type=float,  default=cmfthr,
                        help='PPMM threshold (default=%.2f)'%cmfthr)
    parser.add_argument('--model_version', type=str,  default=model_version,
                        help='MSF model version (default=%s)'%model_version)
    parser.add_argument('--outdir', type=str,  default='.',
                        help='Output path for detection lists')
    parser.add_argument('salience_image', type=str,  
                        help='Salience map image file')
    parser.add_argument('cmf_image', type=str,  
                        help='CMF image file')    

    args = parser.parse_args()

    probthr = args.prob_thr
    cmfthr  = args.ppmm_thr
    outdir  = args.outdir 
    salimgf = args.salience_image
    cmfimgf = args.cmf_image
    modelv  = args.model_version

    salimg,salmm = openimgmm(salimgf)
    cmfimg,cmfmm = openimgmm(cmfimgf)
    if np.argmin(cmfmm.shape) == 0:
        cmfmm = cmfmm.transpose(1,2,0)
    cmfmap = mapinfo(cmfimg,astype=dict)
    cmfbase = basename(cmfimgf)
    cmfspl = cmfbase.split('_')
    cmflid = cmfspl[0]
    outdir = pathjoin(outdir or '.',cmfbase+'_detections')
    detdf = salience2detections(salmm,cmfmm,probthr,cmfthr,cmflid,cmfmap,outdir)

    if len(detdf)>0:
        detstrs = [cmfbase,modelv,'minsal%.2f'%probthr,'minppmm%.1f'%cmfthr]
        detoutf = '_'.join(detstrs).replace('.','p')+'.xlsx'
        save_detections(pathjoin(outdir,detoutf),detdf)
    else:
        print('No plume detections above minsal=%.2f found in %s'%(probthr,salimgf))

    sys.exit(0)
