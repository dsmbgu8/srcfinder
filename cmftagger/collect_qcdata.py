#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Docstring for collect_qcdata.py
Docstrings: http://www.python.org/dev/peps/pep-0257/
"""
from __future__ import absolute_import, division, print_function
from warnings import warn

import sys, os


from util import *

def basename(p):
    return os.path.split(os.path.splitext(p)[0])[1]

raw_input = input


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser('collect_qcdata.py')

    # keyword arguments
    parser.add_argument('-v', '--verbose', action='store_true',
    			help='Enable verbose output') 
    parser.add_argument('--lidassign', type=str, 
    			help='Flightline assigments csv file')
    parser.add_argument('--plumexls', type=str, 
    			help='Plume list') 
    parser.add_argument('--inpath', type=str, help='inpath',
                        default='./output')
    parser.add_argument('--outbase', type=str, help='outbase',
                        default='./{prefix}_qcout_{suffix}.csv')
    

    args = parser.parse_args()
    
    verbose = args.verbose
    qcpath = os.path.abspath(args.inpath)
    lid2uidf = args.lidassign
    plumexls = args.plumexls
    outbase = args.outbase

    lidassign = []
    if lid2uidf:
        lidassign = pd.read_csv(lid2uidf)

    xlsdf,lidxls = [],[]
    if plumexls:
        xlsdf = load_plumedf(plumexls,"cnn_TP_FP","manual_id")
        #xlsdf = xlsdf.loc[~pd.isnull(xlsdf).all(axis=1)]
        #xlsdf[cidcol] = [cid.split('_')[1] for cid in xlsdf[cidcol].values]
        lidxls = np.unique(xlsdf[lidcol].values)

    def summarize(df):
        print(f'Total flightlines: {len(df[lidcol].unique())}')
        print(f'Total candidates: {len(df)}')
        for label in df[labcol].unique():
            nlab = (df[labcol]==label).sum()
            print(f' {label}: {nlab}')

    summarize(xlsdf)
    lidfiltf = pathjoin('config','lidfilters.json')
    lidfilt = load_lidfilters(lidfiltf)
        
    # donepath = ./output/username/submitted/*_{cid,lid}.csv
    donepath = pathjoin(qcpath,'*','submitted')
    cidfiles = sorted(glob.glob(pathjoin(donepath,'*_cid.csv')))
    lidfiles = sorted(glob.glob(pathjoin(donepath,'*_lid.csv')))

    assert len(cidfiles)==len(lidfiles)

    def csv2lid(csvf):
        return basename(csvf).split('_')[0]

    def csv2uid(csvf):
        return csvf.replace(qcpath,'').split('/')[1]

    ciddf,liddf = [],[]
    for cidcsvf,lidcsvf in zip(cidfiles,lidfiles):
        uid,lid = csv2uid(lidcsvf),csv2lid(lidcsvf)
        assert (lid==csv2lid(cidcsvf)) and (uid==csv2uid(cidcsvf))
        ldf = pd.read_csv(lidcsvf)
        cdf = pd.read_csv(cidcsvf)
        ldf[uidcol] = [uid]
        cdf[uidcol] = [uid]*len(cdf)
        cdf[lidcol] = [lid]*len(cdf)        

        liddf.append(ldf)
        ciddf.append(cdf)

    liddf = pd.concat(liddf,axis=0)
    ciddf = pd.concat(ciddf,axis=0)

    prefix = splitext(plumexls)[0]
    liddf.to_csv(outbase.format(prefix=prefix,suffix="lid_all"),index=False)

    donemask = np.isin(lidassign[lidcol].values,liddf[lidcol].values)
    liddone = lidassign.loc[donemask]
    lidtodo = lidassign.loc[~donemask]
        
    nassign = '?' if len(lidassign)==0 else len(np.unique(lidassign[lidcol].values))
    
    # print user counts
    
    for uid,uiddf in liddf.groupby(uidcol):
        uassign = '?'
        if len(lidassign) != 0:
            uassign = np.count_nonzero(lidassign[uidcol]==uid)
        print(f' {uid}: {len(uiddf)} submitted of {uassign} assigned flightlines')

    print(f'Total completed flightlines: {len(np.unique(liddone[lidcol].values))} of {nassign} assigned')
    print(f'Total remaining flightlines: {len(np.unique(lidtodo[lidcol].values))} of {len(lidxls)}')

    import datetime
    dateval = datestr(datetime.datetime.now())
    liddone.to_csv(f'lid2uid_done_{dateval}.csv',index=False)    
    lidtodo.to_csv(f'lid2uid_todo_{dateval}.csv',index=False)

    ciddf[clscol] = np.zeros(len(ciddf),dtype=np.int8)
    ciddf[evalcol] = ['??']*len(ciddf)
    
    if len(xlsdf)!=0:
        lidcid_csv = np.array([lid+'-'+str(cid) for lid,cid in ciddf[[lidcol,cidcol]].values])
        lidcid_xls = np.array([lid+'-'+str(cid) for lid,cid in xlsdf[[lidcol,cidcol]].values])

        xlsdf = xlsdf.loc[np.isin(lidcid_xls,lidcid_csv)]
        
        cidinxls = np.isin(lidcid_csv,lidcid_xls)
        lidcid_xls = lidcid_csv[cidinxls]
        lidcid_new = lidcid_csv[~cidinxls]

        print(f'Candidates in plume list:   {len(lidcid_xls)}')
        print(f'New user-added candidates:  {len(lidcid_new)}')

        cids = ciddf[cidcol].values
        lids = ciddf[lidcol].values
        addcols = [col for col in xlsdf if col not in ciddf]
        for cidx in range(len(xlsdf)):
            cidxls = xlsdf.iloc[cidx]
            cidmsk = (cids==cidxls[cidcol]) & (lids==cidxls[lidcol])
            if not cidmsk.any():
                continue
            ciddf.loc[cidmsk,addcols] = cidxls[addcols].values

    cidallf = outbase.format(prefix=prefix,suffix="cid_all")
    ciddf.to_csv(cidallf,index=False)

    lidfiltf = outbase.format(prefix=prefix,suffix="lid_filt")
    if not pathexists(lidfiltf):
        lidlabels = np.array([keeplab]*len(liddf))
        for col in liddf:
            if col in (lidcol,uidcol):
                continue
            colvals = liddf[col].values
            colaccept = np.isin(colvals,lidfilt[col]['accept'])
            colreject = np.isin(colvals,lidfilt[col]['reject'])

            assert (colaccept | colreject).all()

            lidlabels[colreject] = col


        print(f'counts(lidlabels): {counts(lidlabels)}')
        lidlabels[lidlabels!=keeplab] = rejlab
        liddf[labcol] = lidlabels
        labidx = list(liddf.columns).index(labcol)

        # get flightlines labeled by multiple users with conflicting labels
        uniqcols = [col for col in liddf if col!=uidcol]

        # liduniq = unique rows wrt uniqcols
        uniqrows = liddf.loc[~liddf.duplicated(keep=False,subset=uniqcols)]

        # liddups = unique rows with identical lids (multiple user labels)
        liddups = uniqrows.loc[uniqrows.duplicated(keep=False,subset=lidcol)]

        print(f'liddups: {liddups.shape}')
        print(f'{len(np.unique(liddups[lidcol].values))} flightline label conflicts')

        nliddif = 0
        for lid,labdiff in liddups.groupby(lidcol):
            if len(np.unique(labdiff[labcol].values))==1:
                # label consensus among users, no conflict
                continue
            print(f'{lid}: {len(labdiff)} conflicts')
            nliddif += 1
            coldiff = [col for col in labdiff if len(labdiff[col].unique())>1]

            print(labdiff.loc[:,coldiff])
            lidmask = liddf[lidcol].values==lid
            yn = raw_input('Reject flightline (y/n/m): ').lower()
            if yn.startswith('y'):
                liddf.loc[lidmask,labcol] = rejlab
            elif yn.startswith('n'):
                liddf.loc[lidmask,labcol] = keeplab
            else:
                liddf.loc[lidmask,labcol] = amblab

        print(f'nliddif: {nliddif}')
        liddf.to_csv(lidfiltf,index=False)

    liddf = pd.read_csv(lidfiltf)

    lidreject = liddf.loc[liddf[labcol].values==rejlab,lidcol].values
    print(f'Total rejected flightlines: {len(lidreject)} of {len(liddf)}')

    ciddf.loc[np.isin(ciddf[lidcol].values,lidreject),labcol] = rejlab
    cidreject = ciddf.loc[ciddf[labcol]==rejlab,[lidcol,cidcol]]
    print(f'Total candidates in rejected flightlines: {len(cidreject)} of {len(ciddf)}')

    labidx = list(ciddf.columns).index(labcol)
    clsidx = list(ciddf.columns).index(clscol)

    nrejs = nnegs = nposs = nambs = nrejm = nnegm = nposm = nambm = 0
    nmulti = nsingle = ndif = 0
    namball = nnegall = nposall = nposneg = nambpos = nambneg = nrejdif = 0

    difdf = []

    lidcid_xls = np.array([lid+'-'+str(cid) for lid,cid in xlsdf[[lidcol,cidcol]].values])
            
    # compare multiply labeled candidates in non-rejected flightlines
    for (lid,cid),lidciddf in ciddf.groupby([lidcol,cidcol]):
        cidlabs = np.unique(lidciddf[labcol].values)
        uselab = None
        
        if len(cidlabs)!=1: # users disagree
            isnew = False
            if f'{lid}-{cid}' not in lidcid_xls:
                isnew = True
            
            nmulti += 1
            ndif += 1
            rejcids = cidlabs==rejlab
            ambcids = np.isin(cidlabs,amblabs)
            poscids = np.isin(cidlabs,poslabs)
            negcids = np.isin(cidlabs,neglabs)
            if rejcids.any():
                nrejdif += 1
                uselab = rejlab
            elif ambcids.all():
                nambm += 1
                ndif -= 1                
                namball += 1
                uselab = amblab
            elif negcids.any() and poscids.any():
                if not isnew:
                    difdf.append(lidciddf)
                    nposneg += 1
                uselab = amblab                
            elif ambcids.any():
                if poscids.any():
                    if not isnew:
                        difdf.append(lidciddf)
                        nambpos += 1
                if negcids.any():
                    if not isnew:
                        difdf.append(lidciddf)
                        nambneg += 1 
                uselab = amblab
            elif negcids.all():
                nnegm += 1
                ndif -= 1                
                nnegall += 1
                uselab = falselab
            elif poscids.all():
                nposm += 1
                ndif -= 1
                nposall += 1
                uselab = superlab if superlab in cidlabs else plumelab
            #print(f'{lid}-{cid}: {len(cidlabs)} conflicting labels: {cidlabs} -> {uselab}')
        else: # users agree or single user
            uselab = cidlabs[0]
            if len(lidciddf[labcol].values)!=1:
                nmulti += 1
                if uselab == rejlab:
                    nrejm += 1
                elif uselab in amblabs:
                    nambm += 1
                elif uselab in neglabs:
                    nnegm += 1
                elif uselab in poslabs:
                    nposm += 1                
            else:
                nsingle += 1
                if uselab == rejlab:
                    nrejs += 1
                elif uselab in amblabs:
                    nambs += 1
                elif uselab in neglabs:
                    nnegs += 1
                elif uselab in poslabs:
                    nposs += 1
            #print(f'{lid}-{cid}: {len(cidlabs)} agreeing labels: {uselab}')

        lidmask = ciddf[lidcol].values==lid
        cidmask = lidmask & (ciddf[cidcol].values==cid)
            
        clslab = 0
        if uselab in neglabs:
            clslab = -1
            ciddf.loc[cidmask,evalcol] = 'FP'
        elif uselab in poslabs:
            clslab = 1
            ciddf.loc[cidmask,evalcol] = 'TP'

        ciddf.loc[cidmask,labcol] = uselab
        ciddf.loc[cidmask,clscol] = clslab

    difdf = pd.concat(difdf,axis=0)
    difdf = difdf.sort_values(by=[lidcol,cidcol])
    difoutf = outbase.format(prefix=prefix,suffix="cid_diff")
    difdf.to_csv(difoutf,index=False)
    print('difdf:\n%s'%str((difdf)))

    print(f'nsingle = {nsingle}: nrej = {nrejs}, nneg = {nnegs}, npos = {nposs}, namb = {nambs}')
    print(f'nmulti = {nmulti}:   nrej = {nrejm}, nneg = {nnegm}, npos = {nposm}, namb = {nambm}, ndif = {ndif}')
    print(f'nnegall = {nnegall}, nposall = {nposall}, namball = {namball}')
    print(f'nposneg = {nposneg}, nambpos = {nambpos}, nambneg = {nambneg}, nrejdif = {nrejdif}')
        
    cidfiltf = outbase.format(prefix=prefix,suffix="cid_filt")
    ciddf.to_csv(cidfiltf,index=False)

    ndmask = ~ciddf.duplicated(keep='first',subset=[lidcol,cidcol])
    ndmask = ndmask & (ciddf[labcol].values!=duplab)
    ciddf = ciddf.loc[ndmask]
    cidndf = outbase.format(prefix=prefix,suffix="cid_filt_nodup")
    ciddf.to_csv(cidndf,index=False)

    summarize(ciddf)
    
    ciddf = ciddf.loc[ciddf[labcol].values!=rejlab]
    cidndrf = outbase.format(prefix=prefix,suffix="cid_filt_nodup_norej")
    ciddf.to_csv(cidndrf,index=False)

    ciddf = ciddf.loc[~np.isin(ciddf[labcol].values,amblabs)]
    cidoutf = outbase.format(prefix=prefix,suffix="cid_filt_nodup_norej_noamb")
    ciddf.to_csv(cidoutf,index=False)    
    
    
