#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Docstring for summarize_mask.py
Docstrings: http://www.python.org/dev/peps/pep-0257/
"""
from __future__ import absolute_import, division, print_function
from warnings import warn

import sys, os
import matplotlib
matplotlib.use('Agg')

from skimage.measure import label as imlabel
from pandas import DataFrame, read_csv

from srcfinder_util import *

def center_of_mass(img, labels=None, index=None):    
    from scipy.ndimage.measurements import center_of_mass as _cmass
    return _cmass(img,labels=labels,index=index)

def summarize_obs(obsf):
    '''
    FOVMAX = (ang fov = 36 \pm 2 deg) = 36/2 + 2 = 20 deg
    '''
    ANG_FOVMAX = 20.0
    obslid = filename2flightid(obsf)
    obsimg,obsmm = openimgmm(obsf,writable=False)
    nodatav = float(obsimg.metadata.get('data ignore value',-9999))

    tosensorzenith = obsmm[...,2]
    print('obsmm.shape: "%s"'%str((obsmm.shape)))

    datamask = ~(tosensorzenith==nodatav)
    fovoutl = datamask & (tosensorzenith>ANG_FOVMAX)
    nfovoutl = np.count_nonzero(fovoutl)

    if nfovoutl!=0:
        pathlengths = obsmm[...,0]
        fovinl = datamask & ~fovoutl
        print('nfovoutl,extrema(pathlengths(fovoutl): "%s"'%str((nfovoutl,
                                                                 extrema(pathlengths[fovinl]),
                                                                 extrema(pathlengths[fovoutl]))))            

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser('summarize_cmf.py')

    # keyword arguments 
    parser.add_argument('-v', '--verbose', action='store_true',
    			help='Verbose output')
    parser.add_argument('--robust', action='store_true',
    			help='Use robust statistics')    
    parser.add_argument('-j', '--jobs', type=int, default=1,
    			help='Number of parallel jobs (1 job per image)')    
    parser.add_argument('--plot', action='store_true',
    			help='Plot column statistics')
    parser.add_argument('--randomize', action='store_true',
    			help='Randomize cmffiles processing order')    

    parser.add_argument('--outdir', type=str, default='.',
    			help='Output directory')    

    # positional arguments 
    parser.add_argument('cmffiles', help='CMF image file', type=str,
                        metavar='cmf_file', nargs="+") 

    args = parser.parse_args()

    verbose = args.verbose
    robust = args.robust
    plots = args.plot
    outdir = args.outdir
    n_jobs = args.jobs
    randomize = args.randomize
    cmffiles  = args.cmffiles

    if len(cmffiles)>1 and randomize:
        cmffiles = np.array(cmffiles)[np.random.permutation(len(cmffiles))]

    def loadstats(colcsv):
        return read_csv(colcsv)

    #colcsv = 'cmfstats/ang20150419t230200_cmf_v1f_clip_column_stats.csv'
    #colcsv = 'cmfstats/ang20150419t185922_cmf_v1f_clip_column_stats.csv'
    #colcsv = 'cmfstats/ang20150419t192226_cmf_v1f_clip_column_stats.csv'


    def summarize(cmff,outdir,use_robust_stats=False,plot_stats=False):
        cmfindex = ['# Line name']
        cmfcols = ['percent','npixels','nregions']

        if use_robust_stats:
            statcols = ['npix','med','mad','p05','p95']
        else:
            statcols = ['npix','avg','std','min','max']  
        
        outbase = pathsplit(splitext(cmff)[0])[1]
        colcsv = pathjoin(outdir,outbase+'_column_stats.csv')
        if os.path.exists(colcsv) and not plot_stats:
            print(colcsv,'exists, exiting')
            return False
        print(f'processing {outbase}')

        cmflid = filename2flightid(cmff)
        cmfimg,cmfmm = openimgmm(cmff,writable=False)
        cmfbands = [bc.split()[0] for bc in cmfimg.metadata['band names']]

        nodatav = np.float32(cmfimg.metadata.get('data ignore value',-9999))
        rgb = np.float32(np.clip(cmfmm[...,:-1].copy()/15,0,1))
        cmf = np.float32(cmfmm[...,-1].copy())
        cmfnodata = (cmf==nodatav) | np.isnan(cmf)
        cmfmask = ~cmfnodata & (cmf>0)
        if not cmfmask.any():
            print('all CMF pixels nodata or < 0')
        cmfnvalid = np.count_nonzero(~cmfnodata)
        cmfnmask = np.count_nonzero(cmfmask)
        print(f'CMF # positive={cmfnmask}, valid={cmfnvalid}')

        cmfsummary = [cmflid]

        cmf[~cmfmask] = np.nan
        colnum = np.count_nonzero(cmfmask,axis=0)
        if use_robust_stats:
            colavg = np.nanmedian(cmf,axis=0)
            colstd = np.nanmedian(np.abs(cmf-colavg),axis=0)
            colmin,colmax = extrema(cmf,p=0.95,axis=0)
        else:         
            colavg = np.nanmean(cmf,axis=0)
            colstd = np.nanstd(cmf,axis=0)
            colmin = np.nanmin(cmf,axis=0)
            colmax = np.nanmax(cmf,axis=0)

        
        print('Saving column stats to',colcsv)
        colidx = np.arange(colnum.shape[0])
        coldf = DataFrame(np.c_[colnum,colavg,colstd,colmin,colmax],
                          index=colidx,columns=statcols)
        coldf.to_csv(colcsv,index=False)

        if not plot_stats:
            return True

        import pylab as pl
        maxidx = np.argmax(colavg)
        colfigf = splitext(colcsv)[0]+'.pdf'
        figkw = dict(figsize=(24,3*3.25),sharex=False,sharey=False)
        fig,ax = pl.subplots(3,1,**figkw)
        
        ax[0].imshow(rgb.transpose((1,0,2)))
        ax[0].imshow(cmf.transpose((1,0)),vmin=500,vmax=1500,cmap='YlOrRd',
                     interpolation='none')
        ax[0].set_ylabel('CMF column',size='small')
        ax[0].axhline(maxidx,c='m',ls='--')
        
        ax[1].set_title(outbase)
        ax[1].plot(colidx,colavg,c='b')
        ax[1].plot(colidx,colavg-colstd,c='b',ls='--',alpha=0.5)
        ax[1].plot(colidx,colavg+colstd,c='b',ls='--',alpha=0.5)
        ax[1].axhline(np.mean(colavg)-np.mean(colstd),c='gray',ls='--')
        ax[1].axhline(np.mean(colavg)+np.mean(colstd),c='gray',ls='--')
        ax[1].set_ylabel('CMF $\mu \pm \sigma$ (ppmm)')
        
        ax[2].plot(colidx,100*colnum/rgb.shape[0])
        ax[2].set_ylim(0.0,100.0)
        ax[2].set_ylabel('Valid pixels (%)')
        ax[2].set_xlabel('CMF column')

        for axi in (ax[1],ax[2]):
            axi.set_xlim(0,598)
            axi.axvline(maxidx,c='m',ls='--',alpha=0.8)
            
        pl.tight_layout()
        #pl.show()
        pl.savefig(colfigf)
        pl.close(fig)

        colstats = coldf # loadstats(colcsv)
        print('colstats: "%s"'%str((colstats)))
        colrwinf = splitext(colcsv)[0]+'_rwin.pdf'
        win = 3
        if use_robust_stats:
            colavg = colstats.med
        else:
            colavg = colstats.avg

        
        figkw = dict(figsize=(25,6.5+0.25),sharex=True,sharey=False)
        fig,ax = pl.subplots(2,1,**figkw)
        rwin = colavg.rolling(win,center=True).median()
        rwin[0] = np.nanmedian(colavg.values[:win])
        rwin[-1] = np.nanmedian(colavg.values[-win:])
        coldiff = colavg-rwin
        colsigma = mad(colavg[colavg==colavg])
        print(f'colsigma={colsigma}')

        ax[0].plot(colavg)
        ax[0].plot(rwin)
        ax[1].plot(coldiff)

        c=('yellow','orange','red')
        for i in range(3):
            print(f'{i+1}sigma detections: {np.count_nonzero(coldiff>(i+1)*colsigma)}')
            ax[1].axhline((i+1)*colsigma,c=c[i])

        #ax.plot(colstats.avg.mean()+rwin)#,std=colstats['avg'].std()))
        #ax.plot(colstats.avg.mean()-rwin)#,std=colstats['avg'].std()))
        ax[0].set_xlim(0,598)
        pl.tight_layout()
        pl.savefig(colrwinf)
        pl.close(fig)


        # cmflbl,cmfncc = imlabel(cmfmask.squeeze(),connectivity=2,return_num=True)
        # cmfnpix = np.count_nonzero(cmfmask)
        # cmfpercent = 100*cmfnpix/cmfnvalid
        # cmfsummary.extend([cmfpercent,cmfnpix,cmfncc])
        # cmfindex.extend(['_'.join([bc,col]) for col in cmfcols])

        # #df = DataFrame(cmfsummary,index=cmfbands,columns=['cmfed_percent',
        # #                                                    'cmfed_npixels',                                                        
        # #                                                    'cmfed_nregions'])
        # df = DataFrame(cmfsummary,index=cmfindex).T
        # print('cmf summary:\n%s'%str((df)))

        # df.to_csv(outbase+'_summary.csv',index=False)

        return True

    #obsf='/lustre/ang/y15/raw/bak/ang20150419t230603_rdn_obs_ort'
    #summarize_obs(obsf)
    
    kwargs = dict(use_robust_stats=robust,plot_stats=plots)
    if n_jobs==1 and len(cmffiles)==1:
        summarize(cmffiles[0],outdir,**kwargs)
        sys.exit(0)
    
    from dask.distributed import Client, LocalCluster
    cluster = LocalCluster(n_workers=n_jobs, threads_per_worker=1, processes=True,
                           loop=None, start=None, host=None, ip=None,
                           scheduler_port=0, silence_logs=30, dashboard_address=':31337')
    client = Client(cluster)

    futures = [client.submit(summarize,cmff,outdir,**kwargs)
               for cmff in cmffiles]

    results = client.gather(futures)

    
    sys.exit(0)
