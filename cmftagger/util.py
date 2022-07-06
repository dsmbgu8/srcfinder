import os, sys, glob, argparse, subprocess
import json
import time as tm
from collections import OrderedDict
from socket import gethostname

import pyproj
pyproj.set_use_global_context()
import cartopy.crs as ccrs
from pyproj import CRS,Transformer

from skimage.io import imread, imsave

import dask.array as da

import rasterio as rio
import xarray as xr
#import rioxarray as rxr

import numpy as np, pandas as pd
import matplotlib, matplotlib.pyplot as plt


from bokeh.models import CustomJSHover, HoverTool, CrosshairTool, ResetTool
from bokeh.models.widgets.tables import SelectEditor
from bokeh.models.widgets.tables import NumberFormatter

from datashader.utils import orient_array, ngjit, lnglat_to_meters
import datashader.transfer_functions as xfn

import holoviews as hv, holoviews.operation.datashader as hd, bokeh as bk
hv.extension('bokeh', logo=False)

from holoviews import opts, dim
from skimage.measure import label as imlabel

sys.path.append(os.path.split(__file__)[0]+'/..')
from srcfinder_util import extrema, counts, labpng2tif


nonesel = '[none]'

# plume list xls columns
cmfcol  = 'CMF Image'
evalcol = 'True_pos/false_pos'
lidcol  = 'Line name'
cidcol  = 'Candidate ID'
latcol  = 'Plume Latitude (deg)'
loncol  = 'Plume Longitude (deg)'

# new columns
labcol  = 'Label'
poscol  = 'Positive'
clscol  = 'Class Label'
idxcol  = 'Index'
uidcol  = 'User ID'
donecol = 'QC Completed'
rowcol  = 'Row'
colcol  = 'Col'
xcol    = 'Web Mercator X (m)'
ycol    = 'Web Mercator Y (m)'

keeplab = 'Accept'
rejlab  = 'Reject'


# positive class labels
plumelab = "Plume"
superlab = "Super Plume"
poslabs  = [plumelab,superlab]

# negative class labels
artflab  = "Artifact"
cloudlab = "Cloud"
falselab = "False Enhancement"
neglabs  = [artflab,cloudlab,falselab]

# ambiguous class label
amblab   = "Ambiguous"
bglab    = "Background"
duplab   = "Duplicate"
amblabs  = [amblab,duplab,bglab]

def save_classlabs(jsonoutf):
    cidfilters = OrderedDict()
    for cidclass,labels in zip(("positive","negative","ambiguous"),
                               (poslabs,neglabs,amblabs)):
        cidfilters[cidclass] = labels

    with open(jsonoutf,'w') as fout:
        json.dump(cidfilters,fout)

pathjoin = os.path.join
pathsplit = os.path.split
pathexists = os.path.exists
splitext = os.path.splitext

NODATA = -9999

min_ppmm = 250
max_ppmm = 1500

min_rdn = 0
max_rdn = 15
rdn_lim = (min_rdn,max_rdn)

hires = 'native'

gtif_epsg     = '3857' #'900913' #
gtif_njobs    = 4
gtif_compress = 'LZW'
gtif_dtype    = 'Float32'
gtif_resamp   = 'bilinear'

def glob_re(pattern, strings):
    import re
    return list(filter(re.compile(pattern).match, strings))

#import pytz
#now = lambda: datetime.utcnow().replace(tzinfo=pytz.UTC)
def datestr(date,fmt='%m%d%y'):
    return date.strftime(fmt)

def classlabs(labvals,dtype=np.int8):
    if dtype==str:
        return np.array([plumelab if l.lower().endswith('plume') \
                         else falselab for l in labvals])
    return dtype([l.lower().endswith('plume') for l in labvals])

def next_cid(tgtlab,ciddf,byclass=False):
    tgtcids = ciddf[cidcol].values
    tgtlabv = 1 # assume alphabetical labels for all user candidates
    if byclass:
        labv = classlabs(np.r_[[tgtlab],ciddf[labcol].values])
        tgtlabv,labv = labv[0],labv[1:]
        # new cid = chr(max ascii code + 1) of tgtlab cids
        tgtcids = tgtcids[labv==tgtlabv]
        
    
    if len(tgtcids)==0:
        # positive = alphabetical, negative = integer
        return 'A' if tgtlabv==1 else '1'
    tgtcids = np.array(tgtcids,dtype=str)
    # get next ascii code, convert from int code to char
    # note: assumes if len(cid)>1 -> numeric cid
    maxcode = max([ord(cid) if len(cid)==1 else int(cid)
                   for cid in tgtcids])
    if not byclass and maxcode < ord('A'):
        return 'A'
    return chr(int(maxcode)+1)


def hovertool(img_type,digits=6):
    format_rgba_js = """
        var rgba_bin = (value).toString(2);
        var A = parseInt(rgba_bin.slice(0,8), 2);
        var B = parseInt(rgba_bin.slice(8,16), 2);
        var G = parseInt(rgba_bin.slice(16,24), 2);
        var R = parseInt(rgba_bin.slice(24,32), 2);
        return "" + [R,G,B,A];
    """
    
    format_geo_js = """
        var projections = Bokeh.require("core/util/projections");
        var x = special_vars.x; var y = special_vars.y;
        var coords = projections.wgs84_mercator.invert(x, y);
        var coord = Math.round(coords[{idx}] * 10**{digits}) / 10**{digits};
        return "" + coord.toFixed({digits});
    """

    format_cmf_js = """
        if (value>0)
          return "" + value.toFixed(2);     
        return "[nodata]"
    """

    format_sal_js = """
        if (value>=0) || (value<=1) 
          return "" + (100*value).toFixed(2);
        return "[nodata]"
    """    
    
    format_lab_js ="""
        if (value==255) 
          return "Plume";
        return "[nolabel]"
    """
    

    format_geo_x = format_geo_js.format(idx=0, digits=digits)
    format_geo_y = format_geo_js.format(idx=1, digits=digits)
    formatters = {
        '@x' : CustomJSHover(code=format_geo_x),
        '@y' : CustomJSHover(code=format_geo_y)
    }

    tools = []
    tooltips = [
        ('lat', '@y{custom}'),
        ('lon', '@x{custom}')
    ]

    description = f'{img_type.upper()} Image Hover'
    attachment = 'right'
    if img_type=='cmf':
        tooltips.append(('CMF', '@image{custom}'))        
        formatters['@image'] = CustomJSHover(code=format_cmf_js)
        attachment = 'left'
        #cross = CrosshairTool(dimensions="both",
        #                      line_color='white',
        #                      line_alpha=0.0)
        #tools.append(cross)
    elif img_type=='rgb':
        tooltips.append(('RGB', '@image{custom}'))
        formatters['@image'] = CustomJSHover(code=format_rgba_js)        
    elif img_type=='salience':
        description = f'Salience Map Hover'
        tooltips.append(('Salience', '@image{custom}'))
        formatters['@image'] = CustomJSHover(code=format_sal_js)        
    elif img_type=='labels':
        description = f'Label Map Hover'
        tooltips.append(('LabelMap', '@image{custom}'))
        formatters['@image'] = CustomJSHover(code=format_lab_js)
    elif img_type=='candidates':
        description = f'Candidate Location Hover'
        tooltips = [('CID', '@CID'),(labcol, '@'+labcol)]
    
    hover = HoverTool(name=description,description=description,
                      tooltips=tooltips,formatters=formatters,
                      attachment=attachment,toggleable=True) 
    tools.insert(0,hover)
    
    return tools

def plot_candidates(df,x_range=None,y_range=None):
    #print('plot_candidates x_range,y_range: "%s"'%str((x_range,y_range)))
    seldf  = df[[xcol,ycol,cidcol,labcol]]

    # if None not in (x_range,y_range):
    #     xs,ys = seldf[xcol].values,seldf[ycol].values
    #     mask = ((xs>=x_range[0]) & (xs<=x_range[1]) &
    #             (ys>=y_range[0]) & (ys<=y_range[1]))
    #     seldf = seldf.loc[mask]
    xs,ys,cids,labs = seldf.values.T

    # get index for each label in cidqclabs list to generate a
    # consistent colormapped output
    posstr = classlabs(labs,dtype=str)
    poslab = classlabs(labs,dtype=np.int8)
    cids = np.array(cids,dtype=str)
    points = hv.Points((xs,ys,cids,labs,posstr,poslab),
                       vdims=['CID',labcol,poscol,'Class'])
    labels = hv.Labels((xs,ys,cids,posstr),vdims=['CID',poscol])

    return (points*labels).opts(xlim=x_range,ylim=y_range)

def cmff_to_lid(cmff):
    return os.path.split(cmff)[1].split('_')[0]
  
def meters_to_lnglat(x,y,proj_epsg=gtif_epsg):
    proj_crs = CRS.from_epsg(proj_epsg)
    transformer = Transformer.from_crs(proj_crs, proj_crs.geodetic_crs,
                                       always_xy=True)
    return transformer.transform(x,y)

def meters_to_rowcol(x,y,transform):
    # note: assumes (x,y) are in same projection as transform
    row,col = rio.transform.rowcol(transform, x, y)
    return row,col

def load_plumedf(plumes_file,cnn_sheet,manualid_sheet):
    cnndf = pd.read_excel(plumes_file,sheet_name=cnn_sheet)
    cnndf.columns = cnndf.columns.str.replace('#','').str.strip()
    manualdf = pd.read_excel(plumes_file,sheet_name=manualid_sheet)
    manualdf.columns = manualdf.columns.str.replace('#','').str.strip()
    manualdf.loc[:,evalcol] = ['FN']*len(manualdf)
    plumedf = pd.concat([cnndf,manualdf],axis=0)
    
    #evalcol = plumedf.columns[0]
    #cidcol = plumedf.columns[2]
    labcats = ('FP', 'TP', 'FN')
    ulabcats = plumedf[evalcol].unique()
    if any([val not in labcats for val in ulabcats]):
        print('unexpected labvals: "%s"'%str((ulabcats)))
        raw_input()
    plumedf[cidcol] = [cid.split('-')[-1] for cid in plumedf[cidcol].values]
    plumedf[labcol] = [plumelab if isplumei else falselab
                       for isplumei in np.isin(plumedf[evalcol].values,('TP','FN'))]

    #plumedf = plumedf.sort_values(by=labcol,axis=0)
    return plumedf

def cmf_plumes(df,cmff,cmfxform):
    plumedf = contains_filter(df,cmff_to_lid(cmff),lidcol)
    nplumes = plumedf.shape[0]
    if nplumes==0:
        return plumedf

    # remove rows with nodata values in required columns
    plumedf = plumedf.loc[(plumedf[[loncol,latcol]].values!=-9999.0).all(axis=1)]
    if nplumes != plumedf.shape[0]:
        print(f'Warning: removed {nplumes-plumedf.shape[0]} nodata plumes from plumedf')
    
    # remove rows in same flightline with same (eval_label,lat,lon)
    dedupcols = [lidcol,evalcol,loncol,latcol]
    nplumes = plumedf.shape[0]
    plumedf = plumedf.loc[~plumedf.duplicated(keep='first',subset=dedupcols)]
    if nplumes != plumedf.shape[0]:
        print(f'Warning: removed {nplumes-plumedf.shape[0]} duplicates from plumedf')

    # sort cids first alphabetically (class=1) then numerically (class=(0|1))
    cidsortfn = lambda cids: [ord(v)-ord('0') if len(v)==1 else int(v)
                              for v in cids.values]
    # list positive labs first, then negative
    labsortfn = lambda labs: 1-classlabs(labs.values)
    if len(plumedf) != 0:
        lng,lat = plumedf[loncol].values,plumedf[latcol].values
        x,y = [coord.tolist() for coord in lnglat_to_meters(lng,lat)]
        row,col = meters_to_rowcol(x,y,cmfxform)
        plumedf.loc[:,rowcol] = row
        plumedf.loc[:,colcol] = col
            
        plumedf.loc[:,xcol] = x
        plumedf.loc[:,ycol] = y
        plumedf.loc[:,cidcol] = [cid.split('-')[-1] for cid in plumedf[cidcol].values]
        plumedf = plumedf.sort_values(labcol,key=labsortfn)

    return plumedf # [[loncol,latcol,rowcol,colcol,xcol,ycol,cidcol,labcol]]

def lid2uidassign(plumedf,users,csvoutf='lid2uid_assign.csv'):
    lids = np.unique(plumedf[lidcol].values)
    uids = []
    nlids = len(lids)
    nusers = len(users)
    nulids = int(np.ceil(nlids/nusers))
    print(f'{nlids} total lids: assigning {nulids} lids / user for {nusers} users')
    for uid in users:
        uids.extend([uid]*nulids)
    uids = np.random.permutation(uids)[:nlids]
    df = pd.DataFrame(np.c_[lids,uids],columns=[lidcol,uidcol])
    df.to_csv(csvoutf,index=False)
    print('saved lid2uid df ({}):\n{}'.format(csvoutf,df))
    
def collect_lidqcdat(qcxlsf,lid2userf,username,donepath,filter_uid=False):
    donepaths = glob.glob(pathjoin(donepath,'*.csv'))
    donefiles = set([pathsplit(csvf)[1] for csvf in donepaths])

    imgfiles = set([csvf.replace('_lid.csv','').replace('_cid.csv','')
                    for csvf in donefiles])
    donelids = [cmff_to_lid(imgf) for imgf in imgfiles
                if (imgf+'_cid.csv' in donefiles and
                    imgf+'_lid.csv' in donefiles)]
        
    qcliddf = pd.read_excel(qcxlsf)
    uiddf = pd.read_csv(lid2userf)
    
    qcliddf = qcliddf.loc[:,~pd.isnull(qcliddf.columns)]
    qccols = qcliddf.columns.values    
    qcnotes = qccols[2]
    
    qcdescription = qcliddf.iloc[0].values

    # parse qc column dtype and column values
    # from row 1 of spreadsheet if present
    qcvals,qcdefs = OrderedDict(),OrderedDict()
    if qcliddf.iloc[1,0]=='String':
        vals = qcliddf.iloc[1]
        for c in vals.index:
            if c in (lidcol,cmfcol,uidcol,qcnotes):
                continue
            qcvals[c] = vals[c].split(',')
            qcdefs[c] = qcvals[c][0]
        qcliddf = qcliddf.iloc[2:]
    else:
        qcliddf = qcliddf.iloc[1:]
        
    # add lidcol if only cmfcol present
    if lidcol not in qcliddf and cmfcol in qcliddf:
        qcliddf.loc[:,lidcol] = qcliddf[cmfcol].apply(lambda s: cmff_to_lid(s))

    # populate table with lid -> user assigments
    qcliddf[[lidcol,uidcol]] = uiddf[[lidcol,uidcol]].values

    # fill empty cells with empty strings
    qcliddf = qcliddf.fillna('')
    
    qcxlsbase = pathsplit(splitext(qcxlsf)[0])[1]
    #print(f'{len(qcliddf)} total products listed in {qcxlsbase}')
    if filter_uid:
        qcliddf = contains_filter(qcliddf,username,uidcol)
        #print(f'{len(qcliddf)} products assigned to user {username}')

    # init donecol based on files present in donepath
    qcliddf.loc[:,donecol] = np.isin(qcliddf[lidcol].values,donelids)
    #qc_flags = qccols[~np.isin(qccols,qc_meta)]

    return qcliddf,qcnotes,qcvals,qcdefs

def init_lidfilters(jsonoutf,lidvals):
    lidfilters = OrderedDict()
    for key,value in lidvals.items():
        lidfilters[key] = OrderedDict()
        lidfilters[key]["accept"] = value
        lidfilters[key]["reject"] = []

    with open(jsonoutf,'w') as fout:
        json.dump(lidfilters,fout)

def load_lidfilters(jsonf):
    with open(jsonf,'r') as fid:
        lidfilters = json.load(fid)    

    return lidfilters

def cidexists(ciddf,x,y,label,dthr=0.5):
    # check if cid with label exists within dthr meters of (x,y)
    xydup = ((np.abs(ciddf[xcol].values-x)<dthr) &
             (np.abs(ciddf[ycol].values-y)<dthr))
    if xydup.any():
        dfdup = ciddf.loc[xydup]
        return (dfdup[labcol].values == label).any()
    return False

def contains_filter(df, pattern, column):
    if not pattern:
        return df
    outdf = df[df[column].str.contains(pattern)]    
    return outdf

def img_to_gtif(cmff,scale,gtif_path,epsg=gtif_epsg,compress=gtif_compress,
                masked=True,resampling=gtif_resamp,outdtype=gtif_dtype,
                njobs=gtif_njobs,unlock=False,cache_only=False,cog=False):

    if not os.path.exists(gtif_path):
        os.mkdir(gtif_path)
        
    scaleopts,scalestr = '',''
    if not str(scale).startswith(hires):
        if str(scale).endswith('m'):
            scale = scale[:-1]
        scaleopts = f'-tr {scale} {scale}'
        scalestr = f'_{scale}m'

    # select compression predictor wrt datatype
    floatdtype = any([v in outdtype.lower() for v in ['float','double']])
    predictor = 3 if floatdtype else 2
    
    outf = os.path.split(os.path.splitext(cmff)[0])[1]
    outf = outf + scalestr
    if epsg is not None:
        outf += f'_srs{epsg}'
    if cog:
        outf += f'_cog_LZW'
    elif compress is not None:
        outf += f'_{compress}'
        
    outf = os.path.join(gtif_path,outf+'.tif')

    #print('img_to_gtif: "%s"'%str((cmff,outf,scale,epsg,resampling,
    #                               outdtype,compress,njobs)))
    if not os.path.exists(outf):
        print(f'{outf} not found, generating')

        gtifopts = ''
        #gtifopts += f' -co TILED=YES -co INTERLEAVE=BAND'
        gtifopts += f' -srcnodata {NODATA} -dstnodata {NODATA}'
        gtifopts += f' -co NUM_THREADS={njobs}'
        gtifopts += f' -overwrite -ot {outdtype} -r {resampling}'
        if epsg is not None:
            gtifopts += f' -t_srs EPSG:{epsg}'

        if cog:
            # always compress with LZW when generating a COG
            gtifopts += f' -of COG -co COMPRESS=LZW -co PREDICTOR={predictor}'
        elif compress is not None:
            gtifopts += f' -co COMPRESS={compress} -co PREDICTOR={predictor}'
                    
        gtifopts += ' '+scaleopts

        lockf = outf+'.lock'
        if os.path.exists(lockf) and not unlock:                
            print(f'lock file {lockf} exists, skipping')
            return None

        runcmd  = f'touch {lockf}; '
        runcmd += f'gdalwarp {gtifopts} {cmff} {outf}; '
        runcmd += f'rm -f {lockf}'
        subprocess.call(runcmd,shell=True)

        if not os.path.exists(outf):
            print(f'an error occurred converting {cmff} to {outf}')
            return False

        print(f'successfully generated {outf}')

    if cache_only:
        return True
    
    #dascale = rxr.open_rasterio(outf,masked=masked,chunk='auto')
    #dascale = rxr.open_rasterio(outf,chunk='auto',mask_and_scale=masked)
    #dascale = xr.open_rasterio(outf,chunk='auto',mask_and_scale=masked)
    dascale = xr.open_rasterio(outf,mask_and_scale=masked)
    
    crs = dascale.crs
    #print(f'dascale crs: {crs}')
    if epsg is not None and str(epsg) not in outf:
        print(f'reprojecting {outf} to EPSG:{epsg}')
        raw_input()
        #dascale = dascale.rio.reproject(f'EPSG:{epsg}')
        print(f'dascale (reprojected) crs: {crs}')
    return dascale

def maskoob(da):
    # compute the out-of-bounds mask for a dataarray
    # null pad x,y by \pm 1, label outer nodata roi, crop label x/y by \pm 1
    # oobmask == ccomp value = 1
    mask = np.isin(da.pad(x=(1,1),y=(1,1)).values,da.nodatavals)
    mask = imlabel(mask,connectivity=2)==1
    mask = mask[1:-1,1:-1]

    return mask

def load_cache_gtif(cmff,scale,gtif_path,**kwargs):
    cmfbase = pathsplit(splitext(cmff)[0])[1]
    to = tm.time()
    da = img_to_gtif(cmff,scale,gtif_path,**kwargs)
    print('{} @ {} resolution load time elapsed = {:.3f}s'.format(cmfbase,
                                                              scale,
                                                              tm.time()-to))  
    return da
    
def get_csv(cmfimgf,out_path):
    print('out_path: "%s"'%str((out_path)))
    userdir = os.path.join(out_path,os.environ["USER"])
    if not os.path.exists(userdir):
        os.makedirs(userdir)        
    csvf = os.path.join(userdir,cmfimgf+'_qc.csv')
    print('get_csv:')
    print(f'cmfimgf: {cmfimgf}')
    print(f'userdir: {userdir}')
    print(f'csvfile: {csvf}')
    return csvf

def filter_lids(image_files,keep_lids):
    images_keep = np.array(image_files)
    image_lids = np.array(list(map(cmff_to_lid,image_files)))
    print(f'image_lids: {image_lids}')
    images_keep = images_keep[np.isin(image_lids,keep_lids)]
    print(f'{len(images_keep)} of {len(image_files)} images remain after flightline filtering')
    return images_keep
    
def load_csv(csvf,flag_opts):    
    if csvf is None or not os.path.exists(csvf):
        print(f'load_csv: csvf "{csvf}" does not exist')
        return [],'""'  
    df = pd.read_csv(csvf,delimiter=',').T
    df.columns = ['name','value']
    df.index = df.name.str.strip()
    print('load_csv df:\n%s'%str((df)))
    usernotes = str(df.iloc[-1]['value']).replace('"','')    
    print('usernotes: "%s"'%str((usernotes)))
    keepflags = [flag for flag in flag_opts if flag in df.index.values]
    csvflags = df.loc[keepflags]
    userflags = csvflags.index[csvflags.loc[keepflags,'value']=='1'].values
    print('userflags: "%s"'%str((userflags)))
    return userflags,usernotes

def save_csv(csvf,flags,flag_opts,usernotes=""):
    if csvf is None or not os.path.exists(csvf):
        print(f'save_csv: csvf "{csvf}" does not exist')
    csvout = [(flag.strip(),int(flag in flags)) for flag in flag_opts]
    usernotes = usernotes.replace('"','')
    csvout.append(('notes',f'"{usernotes}"'))
    df = pd.DataFrame(csvout,columns=['name','value']).T
    df.to_csv(csvf,index=False)
    assert((load_csv(csvf)[0]==flags).all())


def merge_rgb(fgrgb,bgrgb):
    return fgrgb.where(~fgrgb.any(axis=0),bgrgb)

@ngjit
def rdn2uint8(agg,rdn_lim):
    out = np.zeros_like(agg,dtype=np.uint8)
    min_rdn,max_rdn = rdn_lim[0],rdn_lim[1]
    rdn_range = max_rdn - min_rdn
    for i in range(agg.shape[0]):
        for j in range(agg.shape[1]):
            norm = (agg[i,j] - min_rdn) / rdn_range
            #norm = 1 / (1 + np.exp(c * (th - norm))) # bonus
            out[i,j] = np.uint8(min(max(0,norm),1) * 255)
    return out

def cmf2rgba(agg,cmap,cmf_lim):
    out = np.zeros([agg.shape[0],agg.shape[1],4],dtype=np.uint8)
    norm = (orient_array(agg)-cmf_lim[0])/(cmf_lim[1]-cmf_lim[0])
    print('norm: "%s"'%str((norm)))
    out[...,:3] = 255*cmap(np.clip(norm,0,1))[...,:3]
    out[...,-1] = np.where(np.isnan(norm) | (norm<=0),0,255)
    return out

@ngjit
def mergergba(rgba_a,rgba_b):
    # a over b
    assert((rgba_a.shape[0]==4) and (rgba_b.shape[0]==4))
    assert((rgba_a.dtype==float) and (rgba_b.dtype==float))
    assert((rgba_a.min()>=0) and (rgba_a.max()<=1))
    assert((rgba_b.min()>=0) and (rgba_b.max()<=1))
    out = np.zeros_like(a,dtype=np.float32)
    rows,cols = rgba_a.shape[1:]
    for i in range(rows):
        for j in range(cols):
            out[-1,i,j] = rgb_a[-1,i,j] + rgba_b[-1,i,j]*(1-rgba_a[-1,i,j])
            out[:3,i,j] = rgb_a[:,i,j]*rgba_a[-1,i,j] + \
                          rgb_b[:,i,j]*rgba_b[-1,i,j]*(1-rgba_a[-1,i,j])
            out[:3,i,j] = np.uint8(out[:,i,j]*out[-1,i,j])
    return out


def combine_bands(rgb, rgb_lim, xs=[], ys=[]):
    xs = xs if len(xs)!=0 else rgb['x']
    ys = ys if len(ys)!=0 else rgb['y'][::-1]
    try:
        r, g, b = [orient_array(img) for img in rgb]
        a = (np.where((np.isnan(r) | (r<=(NODATA+1))),0,255)).astype(np.uint8)    
    except:
        r,g,b,a = rgb

    r, g, b = [rdn2uint8(img, rgb_lim) for img in (r,g,b)]

    return hv.RGB((xs, ys, r, g, b, a), vdims=list('RGBA'), kdims=['x','y'])

interpolation='bilinear' # 'nearest'
def regrid(iview,interpolation=interpolation):
    gridkw = dict(interpolation=interpolation)
    return hd.rasterize(iview,**gridkw)

def imagebounds(img):
    xbounds = extrema(img.x.values)
    ybounds = extrema(img.y.values)
    return xbounds,ybounds

def framebounds(data,frameaspect,buf=0.05):
    img = None
    if isinstance(data,dict):
        for key in ('cmf','rgb'):
            if key in data:
                img = data[key]
                break
    else:
        img = data
        
    if img is None:
        return None,None

    xbounds,ybounds = imagebounds(img)

    xdiff = xbounds[1]-xbounds[0]
    ydiff = ybounds[1]-ybounds[0]
    xmid = xbounds[0]+xdiff/2
    ymid = ybounds[0]+ydiff/2

    xybuf = 1+buf
    if ydiff > xdiff:
        # hold ybounds fixed, pad xbounds
        xpad = (xybuf*(xdiff*frameaspect))/2
        ypad = (xybuf*ydiff)/2
    else:
        # hold xbounds fixed, pad ybounds
        xpad = (xybuf*xdiff)/2
        ypad = (xybuf*(ydiff*(1/frameaspect)))/2

    xbounds = xmid-xpad, xmid+xpad
    ybounds = ymid-ypad, ymid+ypad

    return xbounds, ybounds


def set_bounds(fig, element):
    print('element: "%s"'%str((element)))
    print('fig.state.x_range.bounds: "%s"'%str((fig.state.x_range.bounds)))
    print('fig.state.y_range.bounds: "%s"'%str((fig.state.y_range.bounds)))

if __name__ == '__main__':
    cids = '2,4,5,A,F,C,1,3,D'.split(',')
    labs = [plumelab,plumelab,plumelab,superlab,plumelab,plumelab,
            duplab,duplab,superlab]
    testdf = pd.DataFrame(np.c_[cids,labs][:5],columns=[cidcol,labcol])

    print('next_cid(amblab,testdf): "%s"'%str((next_cid(amblab,testdf))))
    print('next_cid(plumelab,testdf): "%s"'%str((next_cid(plumelab,testdf))))
