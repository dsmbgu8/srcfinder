from __future__ import absolute_import, division, print_function

import sys,os
import numpy as np
import time

from os.path import join as pathjoin, exists as pathexists, split as pathsplit
from os.path import splitext, abspath, getsize, realpath, isabs, expandvars
from os.path import expanduser
from collections import OrderedDict

from glob import glob


try:
    import gdal
    from gdal import gdalconst, ogr, osr
except:
    from osgeo import gdal
    from osgeo.gdal import gdalconst, ogr, osr

import rasterio as rio
    
from warnings import filterwarnings, warn

from spectral.io.envi import open as envi_open_file
from LatLongUTMconversion import UTMtoLL, LLtoUTM

import geopandas as gpd

from scipy.ndimage.measurements import center_of_mass

# avoid PIL decompression bomb exception for huge images
from PIL import Image
Image.MAX_IMAGE_PIXELS = 1000000000

warnings_ignore = [
    'low contrast image', 
]

for msg in warnings_ignore:
    filterwarnings("ignore", message='.*%s.*'%msg)

def _raw_input(*args):
    try:
        return input(*args)
    except KeyboardInterrupt as ki:
        print(ki,file=sys.stderr)
        sys.exit(0)
    except Exception as e:
        print(e,file=sys.stderr)        
        pass

# make python2 and python3 use the same pickle/input/raw_input functions
if sys.version_info[0] == 2:
    import cPickle as pickle
    input = raw_input
elif sys.version_info[0] == 3:
    import pickle # cPickle = pickle in python3
    raw_input = _raw_input
    xrange = range
    _map = map
    map = lambda f,a: list(_map(f,a))

SRCFINDER_ROOT = os.getenv('SRCFINDER_ROOT') or pathsplit(__file__)[0]
SRCFINDER_ROOT = os.path.abspath(SRCFINDER_ROOT)
print('SRCFINDER_ROOT: "%s"'%str((SRCFINDER_ROOT)))
sys.path.append(SRCFINDER_ROOT)
    
mkdir = os.makedirs
mkdirs = mkdir
gettime = time.time

# WGS-84
DATUM_WGS84 = 23 
EPSG_WGS84 = 4326
DEG2RAD = np.pi/180.0

NODATA = -9999
    
# reused constants
ORDER_NN    = 0
ORDER_BILIN = 1
ORDER_QUAD  = 2
ORDER_CUBIC = 3

CONN4 = 1
CONN8 = 2

CMFBG = 0
POINTSRC = 1
DIFFSRC = 2
FALSESRC = 3
LOCSRC = 4 # indicates pixel location of source

posrgb = (255,0,0)
negrgb = (0,255,255)
bgrgb = (255,255,0)
locrgb = (255,255,255)


CMFLABELS = [CMFBG,POINTSRC,DIFFSRC,FALSESRC]

use_absmf=False

kernel=50
mfmin,mfmax = 500,1500
minarea=9
mfminsmall  = 1250

# HDF compression settings
COMPLEVEL = 9      # compression level 9 = most compression
COMPLIB   = 'blosc'  # one of 'blosc', 'zlib','bzip2', 'lzo' (all lossless)

# HDF compression settings
hdfkw = dict(complevel=COMPLEVEL,complib=COMPLIB,mode='r')

# csv/xls columns in plume lists (note: not always lower case)
lidcol,cidcol = 'Line name','Candidate ID'
latcol,loncol = 'Plume Latitude (deg)','Plume Longitude (deg)'
labcol,xlscol = 'Class label','XLS file'

# labimg suffix
labimgsuf = '_mask.png'

def filename(path):
    '''
    /path/to/file.ext -> file.ext
    '''
    return pathsplit(path)[1]

def fileext(path):
    '''
    /path/to/file.ext -> ext
    '''
    return splitext(path)[1]

def dirname(path):
    '''
    /path/to/file.ext -> /path/to
    '''
    return pathsplit(path)[0]

def basename(path):
    '''
    /path/to/file.ext -> file
    '''
    return splitext(filename(path))[0]

def getlatloncols(df):
    latkw=['latitude', ' lat', '(lat', '_lat']
    lonkw=['longitude',' lon', '(lon', '_lon']
    latcols,loncols = [],[]
    for col in df.columns:
        cq = col.lower()
        if any([kw in cq for kw in latkw]):
            latcols.append(col)
        elif any([kw in cq for kw in lonkw]):
            loncols.append(col)

    if len(latcols)==0:
        raise Exception('Latitude column not found')
    elif len(loncols)==0:
        raise Exception('Longitude column not found')

    latcols = sorted(latcols)
    loncols = sorted(loncols)
    
    if len(loncols)!=1 or len(latcols)!=1:
        print('Multiple latitude or longitude columns found:',
              str(latcols),',',str(loncols))
        
        print('Using:',latcols[0],'and',loncols[0])
        #raw_input() #TODO Brian that crashes the program for 2018 CalCH4

    return latcols[0],loncols[0]

def extract_shapes(imgfile,shpfile):
    import rasterio as rio
    import fiona as fio
    from rasterio.mask import mask

    with rio.open(imgfile,'r'),fio.open(shpfile,'r') as img,shp:
        nfeat = len(shp)
        shp_img, shp_t = [None]*nfeat,[None]*nfeat
        for i,feati in enumerate(shp):
            shp_img[i], shp_t[i] = mask(img, feati["geometry"], crop=True)

    return shp_img, shp_t

def df2gdf(df,xcol=loncol,ycol=latcol,epsg=EPSG_WGS84):
    from shapely.geometry import Point
    # http://www.spatialreference.org/ref/epsg/{epsg}
    crs = {'init': 'epsg:'+str(epsg)} 
    geometry = [Point(xy) for xy in zip(df[xcol], df[ycol])]
    return gpd.GeoDataFrame(df, crs=crs, geometry=geometry)

def gdf2file(outf,gdf,**kwargs):
    import fiona
    driver = kwargs.get('driver','ESRI Shapefile')

    if driver == 'KML':
        # Enable fiona KML driver
        gpd.io.file.fiona.drvsupport.supported_drivers['KML'] = 'rw'
    else:
        assert (driver in gpd.io.file.fiona.drvsupport.supported_drivers)

    # Write file
    with fiona.Env():
        # Might throw a WARNING - CPLE_NotSupported in b'dataset sample_out.kml does not support layer creation option ENCODING'
        gdf.to_file(outf, driver=driver)
        print('saved geodataframe to file: "%s"'%str((outf)))
        
        # Make sure we can re-read file
        # return gpd.read_file(outf, driver=driver)
        # print('geo_df from %s:\n%s'%(driver,geo_df))

def df2hdf(hdffile,df,**kwargs):
    key = kwargs.pop('key',basename(hdffile))
    kwargs.setdefault('complib',COMPLIB)
    kwargs.setdefault('complevel',COMPLEVEL)
    return df.to_hdf(hdffile,key,**kwargs)

def hdf2df(hdffile,**kwargs):
    from pandas.io.pytables import HDFStore
    store = HDFStore(hdffile)
    keys = store.keys()
    if 'key' not in kwargs:
        key = keys[0]
        if len(keys)>1:
            warn('Store "%s" contains multiple keys,'
                 ' using key="%s"'%(hdffile,key))

    return store[key]

def imsave(fname,img,**kwargs):
    from skimage.io import imsave as _imsave    
    if kwargs.pop('verbose',False):
        print('saving',fname)
    return _imsave(fname,img,**kwargs)

def array2rgba(a,**kwargs):
    '''
    converts a 1d array into a set of rgba values according to
    the default or user-provided colormap
    '''
    from pylab import rcParams
    from matplotlib.cm import get_cmap
    from numpy import isnan,clip,uint8,where
    rgba = np.zeros(list(a.shape)+[4],dtype=uint8)
    cmap = kwargs.get('cmap',rcParams['image.cmap'])
    lut = kwargs.get('lut',rcParams['image.lut'])
    astuple = kwargs.pop('astuple',False)
    cm = get_cmap(cmap,lut)
    aflat = np.float32(a.copy().ravel())
    nanmask = isnan(aflat)
    if not nanmask.all():
        vmin = float(kwargs.pop('vmin',aflat.min()))
        vmax = float(kwargs.pop('vmax',aflat.max()))
        if vmin<vmax:
            aflat[nanmask] = vmin # use vmin to map to zero+below
            aflat = clip(((aflat-vmin)/(vmax-vmin)),0.,1.)
            rgba = uint8(cm(aflat)*255)
            if nanmask.any():
                nanr = where(nanmask)
                rgba[nanr[0],:] = 0
            rgba = rgba.reshape(list(a.shape)+[4])
        
    if astuple:
        rgba = [tuple(list(rgbai)) for rgbai in rgba]
    
    return rgba

def float2rgba(img,cmap='binary',vmin=0.0,vmax=1.0,alpha=0):
    """
    float2rgba(img,alpha=0)
    
    Summary: Stretch range of unit-scaled 32-bit single band float image to
      4-band (3x8 rgb + 1x8 alpha) uint8 rgba image
    
    Arguments:
    - img: [n,m,1] single band 32-bit float image, pixel values \in [0.0,1.0]
    - alpha: output alpha value (default=0)
    
    Output:
    - [n,m,4] rgba uint8 image with img encoded as 3x8-bit rgb bands and
      constant alpha band

    Notes:
    - img must be scaled to [0,1] range, but should *not* be scaled into
      [FLT_MIN,FLT_MAX] (typically=[1.175494e-38,3.402823e+38]) range
    - most image analysis software ignores alpha band, so we only use the 24-bit
      range instead of the full 32-bit range
    - default alpha=0 will produce transparent images, set alpha=255 or ignore
      alpha band to visualize output
    
    """
    
    # assume img pixel values \in [0,1] range
    assert((img.min()>=vmin) & (img.max()<=vmax))
    if cmap=='binary':
        # max value of uint8 rgb (8x3 bands=24bits) pixel = 2**(24)-1
        rgbavec = np.uint32(((2**24)-1)*img).view(dtype=np.uint8)
    else:
        rgbavec = np.uint8(array2rgba(np.float32(img.ravel()),cmap=cmap,
                                      vmin=vmin,vmax=vmax,lut=4096))
    rgba = rgbavec.reshape([img.shape[0],img.shape[1],4])    
    rgba[...,-1] = np.uint8(alpha)
    return rgba

def rgba2float(img,cmap='binary',alpha=0):
    from pylab import rcParams
    from matplotlib.cm import get_cmap
    imgc = img.copy()
    if cmap=='binary':
        imgc[...,-1] = np.uint8(alpha)    
        imgc = imgc.view(np.uint32) / np.float32((2**24)-1)        
        return imgc.squeeze()
    cm = get_cmap(cmap,lut=4096)    
    lut = cm.colors[...,:3]
    d = ((imgc[...,:3]/255.0-lut[:, None, None, :])**2).sum(axis=-1)
    f = d.argmin(axis=0)/np.float32(len(lut)-1)
    return f # np.round(len(lut)*f))/float(len(lut))

def rgbdet2ql(rgb,det,detmask=[]):
    rgbimg = np.uint8(255*rgb.copy())
    if len(detmask)!=0 and detmask.any():
        ch4idx = np.where(detmask)
        ch4rgba = array2rgba(det[ch4idx],cmap='YlOrRd')
        rgbimg[ch4idx[0],ch4idx[1],:3] = ch4rgba[:,:3]
    return rgbimg #.transpose((1,0,2))[::-1])

def progressbar(caption,maxval=None):
    """
    progress(title,maxval=None)
    
    Summary: progress bar wrapper
    
    Arguments:
    - caption: progress bar caption
    - maxval: maximum value
    
    Keyword Arguments:
    None 
    
    Output:
    - progress bar instance (pbar.update(i) to step, pbar.finish() to close)
    """
    
    from progressbar import ProgressBar, Bar, UnknownLength
    from progressbar import Percentage, Counter, ETA

    capstr = caption + ': ' if caption else ''
    if maxval is not None:    
        widgets = [capstr, Percentage(), ' ', Bar('='), ' ', ETA()]
    else:
        maxval = UnknownLength
        widgets = [capstr, Counter(), ' ', Bar('=')]

    return ProgressBar(widgets=widgets, maxval=maxval)    

def labfillholes(labimg,**kwargs):
    from scipy.ndimage.morphology import binary_fill_holes as binfillholes
    labfilled = labimg.copy()
    for l in np.unique(labimg):
        if l==0:
            continue
        labfilled[binfillholes(labimg==l)] = l

    return labfilled

def findboundaries(labimg,**kwargs):
    from skimage.segmentation import find_boundaries
    kwargs.setdefault('connectivity',CONN8)
    return find_boundaries(labimg,**kwargs)

def thickboundaries(labimg):
    return findboundaries(labimg,mode='thick')

def innerboundaries(labimg):
    return findboundaries(labimg,mode='inner')

def outerboundaries(labimg):
    return findboundaries(labimg,mode='outer')

def createimg(hdrf,metadata,**kwargs):
    from spectral.io.envi import create_image
    return create_image(hdrf, metadata, ext='', force=True)

def imlabel(img,**kwargs):
    from skimage.measure import label as _label
    kwargs.setdefault('connectivity',CONN8)
    return _label(img,**kwargs)

def findobj(labimg,max_label=0):
    from scipy.ndimage.measurements import find_objects as _findobj
    return _findobj(labimg,max_label=max_label)

def disk(radius,**kwargs):
    from skimage.morphology import disk as _disk
    return _disk(radius,**kwargs)

def bwopen(bwimg,selem=disk(3),**kwargs):
    from skimage.morphology import binary_opening as _bwo
    return _bwo(bwimg,selem=selem,**kwargs)

def bwdilate(bwimg,**kwargs):
    from skimage.morphology import binary_dilation as _bwd
    kwargs.setdefault('selem',disk(3))
    return _bwd(bwimg,**kwargs)

def bwdist(bwimg,**kwargs):
    metric = kwargs.get('metric','euclidean')
    if metric=='euclidean':
        kwargs.pop('metric',None) # metric is only option for edt
        from scipy.ndimage.morphology import distance_transform_edt as _bwdist
    elif metric in ('chessboard','taxicab'):
        from scipy.ndimage.morphology import distance_transform_cdt as _bwdist
    kwargs.setdefault('return_distances',True)
    kwargs.setdefault('return_indices',False)
    return _bwdist(bwimg,**kwargs)

def mergelabels(labimg,mergedist,return_merged=False,doplot=False):
    # merge labeled regions <= mergedist pixels from each other
    labmask = labimg!=0
    mergereg = imlabel(bwdist(~labmask,metric='chessboard')<=mergedist)
    mergelab = np.unique(mergereg)[1:]
    mergeimg = np.zeros_like(labimg)
    mergemap = {}
    nbefore = 0
    for mlab,mobj in zip(mergelab,findobj(mergereg)):
        mlmask = (mergereg[mobj]==mlab) & labmask[mobj]
        mergeimg[mobj][mlmask] = mlab
        if return_merged:
            mergemap[mlab] = np.unique(labimg[mobj][mlmask])
            nbefore += len(mergemap[mlab]) 
    print(f'{nbefore} input labels -> {len(mergelab)} merged labels @ mergedist {mergedist}px')
    if doplot:
        import pylab as pl
        fig,ax = pl.subplots(1,2,sharex=True,sharey=True)
        ax[0].imshow(labimg)
        ax[0].set_title(f'{nbefore} labels before')
        ax[1].imshow(mergeimg)
        ax[1].set_title(f'{len(mergelab)} labels after')
        pl.show()
    if return_merged:
        return mergeimg, mergemap
    return mergeimg

def maskbounds(mask,xvals,yvals,plot=True):
    y,x = np.meshgrid(yvals,xvals)
    #y,x = np.meshgrid(np.arange(da.shape[0]),np.arange(da.shape[1]))
    y,x = np.float32(y).T,np.float32(x).T
    x[mask] = y[mask] = np.nan
    #midy,midx = np.nanmean(y,axis=0),np.nanmean(x,axis=1)
    miny,minx = np.nanmin(y,axis=0),np.nanmin(x,axis=1)
    maxy,maxx = np.nanmax(y,axis=0),np.nanmax(x,axis=1)
    midy,midx = miny+(maxy-miny)/2, minx+(maxx-minx)/2
    if plot:
        import pylab as pl
        figkw = dict(figsize=(1*7.5,1*7.5+0.25),sharex=True,sharey=True)
        fig,ax = pl.subplots(1,1,**figkw)
        ax.imshow(mask)
        for i,yi in enumerate((miny,midy,maxy)):
            ax.plot(xvals,yi,c='r')
        for i,xi in enumerate((minx,midx,maxx)):
            ax.plot(xi,yvals,c='b',ls=':')            
        pl.tight_layout()
        pl.show()

    output = dict(x=x,y=y,midx=midx,midy=midy,
                  minx=minx,miny=miny,maxx=maxx,maxy=maxy)        
    return output

def mergedist_mask(img,locs,mergedists):
    mask_shape = img.shape[:2]
    imgmap = mapinfo(img)
    mdimg = np.zeros(mask_shape,dtype=np.float)
    for mdi,mdv in enumerate(mergedists):
        mdsub = disk(int(np.ceil(2*mdv/(2*imgmap['xps']))))
        for lat,lon in locs:
            s,l = latlon2sl(lat,lon,mapinfo=imgmap)
            smin,smax = int(s-mdrad),int(s+mdrad)+1
            lmin,lmax = int(l-mdrad),int(l+mdrad)+1
            mduse = mdimg[lmin:lmax,smin:smax]
            mdimg[lmin:lmax,smin:smax] = np.where(mdsub!=0,mdsub*mdv,mduse)
    return mdimg

def imread(fname,**kwargs):
    from skimage.io import imread as _imread
    kwargs.setdefault('plugin',None)
    return _imread(fname,**kwargs)

def imresize(img,output_shape,**kwargs):
    from skimage.transform import resize as _imresize
    
    kwargs.setdefault('order',ORDER_NN) 
    kwargs.setdefault('clip',False)
    kwargs.setdefault('preserve_range',True)
    if kwargs.pop('anti_alias',False):
        from scipy import ndimage as ndi
        cval = kwargs.get('cval',0)
        mode = kwargs.get('mode','constant')
        sigma = kwargs.pop('anti_alias_sigma',None)
        if sigma is None:
            factors = (np.asarray(img.shape, dtype=float) /
                       np.asarray(output_shape, dtype=float))
            sigma = np.maximum(0, (factors - 1) / 2)
        imgrs = ndi.gaussian_filter(img, sigma, cval=cval, mode=mode)
    else:
        imgrs = img
    
    return _imresize(imgrs,output_shape,**kwargs)

def filename2flightid(filename):
    '''
    get flight id from filename
    ang20160922t184215_cmf
    _v1g_img -> ang20160922t184215
    '''
    return basename(filename).split('_')[0]


def filename2flightyid(filename, dtype=str):
    '''
    get flight id from filename
    ang20160922t184215_cmf_v1g_img -> y16
    '''

    Y, m, d = filename2flightdate(filename, dtype=str)
    return 'y' + Y[-2:]

def filename2flightdate(filename,dtype=str):
    '''
    get flight id from filename
    ang20160922t184215_cmf_v1g_img -> ang20160922t184215
    '''
    
    flightid = filename2flightid(filename)
    if flightid.startswith('f'): # avcl
        datestr = flightid.split('t')[0][1:7]
        Y,m,d = '20'+datestr[:2],datestr[2:4],datestr[4:6]
    else: # ang/prism
        datestr = flightid.split('t')[0][-8:]
        Y,m,d = datestr[:4],datestr[4:6],datestr[6:]
    if dtype!=str:
        Y,m,d = list(map(dtype,[Y,m,d]))
    return Y,m,d

def filename2flighttime(filename,dtype=str):
    '''
    get flight id from filename
    ang20160922t184215_cmf_v1g_img -> ang20160922t184215
    '''
    flightid = filename2flightid(filename)
    if flightid.startswith('f'): # avcl
        timestr = flightid.split('t')[1][:6]
        H,M,S = '20'+timestr[:2],timestr[2:4],timestr[4:6]
    else:
        timestr = flightid.split('t')[1][:6]
        H,M,S = timestr[:2],timestr[2:4],timestr[4:]
    if dtype!=str:
        H,M,S = list(map(dtype,[H,M,S]))
    return H,M,S

def filename2datetime(filename):
    Y,m,d   = filename2flightdate(filename,dtype=int)
    H,M,S   = filename2flighttime(filename,dtype=int)
    datestr = '{m:02d}/{d:02d}/{Y:04d}'.format(**locals())
    timestr = '{H:02d}:{M:02d}:{S:02d}'.format(**locals())
    tstruct = strptime(' '.join([datestr,timestr]),'%m/%d/%Y %H:%M:%S')
    dt = datetime.fromtimestamp(time.mktime(tstruct))
    return dt

def filename2lustre(filename,filepath=True,product='',calid='',
                    suffix='',ort=True):
    from glob import glob
    _,filebase = pathsplit(filename)
    lid = filename2flightid(filebase)
    if product=='' and '_' in filebase:
        fileparts = filebase.split('_')
        product = fileparts[1]
        if len(fileparts)>2:
            calid = fileparts[2]

    yyyy,_,__ = filename2flightdate(filebase,dtype=str)
    year = 'y'+yyyy[-2:]
    if lid[:3] == 'ang':
        platform = 'ang'
    elif lid[:3] == 'prm':
        platform = 'prism'
    else:
        platform = 'avcl'
    path = pathjoin('/lustre',platform,year)
    if len(product)!=0:
        path = pathjoin(path,product)
    if ort:
        path = pathjoin(path,'ort')
        if len(suffix)==0:
            suffix = 'img'

    if filepath:
        fpath = pathjoin(path,filebase)
        if pathexists(fpath):
            return fpath
        fpatn = pathjoin(path,lid+'_*'+product+'_*'+calid+'_*'+suffix)        
        files = glob(fpatn)
        if len(files)!=0:
            return files[0]

    return path

def shortstr(s,width=20,placeholder='...',quote=False):
    shortstr = s if len(s) <= width else s[:width]+placeholder
    return '"%s"'%shortstr if quote else shortstr

def filename2calid(infile):
    # e.g., ./ort/ang20160915t194328_cmf_v1n2_img -> v1n2
    _,filen = pathsplit(infile)
    spl = filen.split('_')
    if filen.startswith('f'): # avcl
        calid = str(spl[1]+'_'+spl[2])
    else:
        calid = spl[2]
    return calid
l1calid = filename2calid

def filename2productid(filename):
    '''
    get product id from filename
    ang20160922t184215_cmf_v1g_img -> cmf
    '''
    return basename(filename).split('_')[1]

def counts(a,sort=True):
    c = OrderedDict()
    uvals,unums = np.unique(a,return_counts=True)
    ncz = zip(unums,uvals)
    if sort:
        ncz = sorted(ncz)
    for num,val in ncz:
        c[val] = num
    return c

def extrema(a,p=1.0,buf=0.0,axis=None,**kwargs):
    if p==1.0:
        vmin,vmax = np.nanmin(a,**kwargs),np.nanmax(a,**kwargs)
    else:        
        assert(p>0.0 and p<1.0)
        apercent = lambda q: np.nanpercentile(a,axis=axis,q=q,interpolation='nearest')
        vmin,vmax = apercent((1-p)*100),apercent(p*100)

    if buf !=0:
        vbuf = (vmax-vmin)*buf
        vmin,vmax = vmin-vbuf,vmax+vbuf
    return vmin,vmax

def utmzone2epsg(zone,hemi):
    assert(hemi in ('N','S'))
    return int(('326' if hemi=='N' else '327') + '%02d'%zone)

def epsg2utmzone(epsg):
    assert (epsg>=32600 and epsg<=32661) or (epsg>=32700 and epsg<=32761) 
    zone = epsg%(epsg//100)
    hemi = 'N' if epsg<32700 else 'S'
    return (zone,hemi)
    
def geo2utmzone(longitude,latitude):
    zone = int(1+(longitude+180.0)/6.0)
    hemi = 'N' if (latitude >= 0.0) else 'S'
    return zone,hemi

def envitypecode(np_dtype):
    from numpy import dtype
    from spectral.io.envi import dtype_to_envi
    _dtype = dtype(np_dtype).char
    return dtype_to_envi[_dtype]

def pixbox(i,j,ijoff,shape,as_slice=False):
    ''' returns indices for a square centered at i, j, with width 2ijoff
    :param i:
    :param j:
    :param ijoff:
    :param shape:
    :param as_slice:
    :return:
    '''
    if isinstance(ijoff,tuple):
        ioff,joff = ijoff
    else:
        ioff = joff = ijoff
    
    imin,jmin = max(0,i-ioff),max(0,j-joff)
    imax,jmax = i+ioff+1,j+joff+1
    if len(shape)>=2:
        imax,jmax = min(imax,shape[0]),min(jmax,shape[1])

    if as_slice:
        return slice(imin,imax),slice(jmin,jmax)
    return imin,imax,jmin,jmax

def inbbox(ij,shape,ijmin=(0,0)):
    i,j = ij
    return (i>=ijmin[0] and i<shape[0]) and (j>=ijmin[1] and j<shape[1])

def extract_tile(img,ul,tdim,verbose=False,transpose=None,fill_value=0):
    '''
    extract a tile of dims (tdim,tdim,bands) offset from upper-left 
    coordinate ul in img, zero pads when tile overlaps image extent 
    '''
    try:
        if len(tdim)==1:
            tdim = (tdim[0],tdim[0])
    except:
        tdim = (tdim,tdim)

    assert (len(tdim)==2)
            
    ndim = img.ndim
    if ndim==3:
        nr,nc,nb = img.shape
    elif ndim==2:
        nr,nc = img.shape
        nb = 1
    else:
        raise Exception('invalid number of image dims %d'%ndim)

    lr = (ul[0]+tdim[0],ul[1]+tdim[1])

    # get absolute offsets into img (clip if outside image extent)
    ibeg,iend = max(0,ul[0]),min(nr,lr[0])
    jbeg,jend = max(0,ul[1]),min(nc,lr[1])

    # get relative offsets into imgtile (shift if outside image extent)
    padt,padl = max(0,-ul[0]), max(0,-ul[1])
    padb,padr = padt+(iend-ibeg), padl+(jend-jbeg)
    
    if verbose:
        print('img.nrows,img.ncols',nr,nc)
        print('ul,lr,tdim',ul,lr,tdim)
        print('padt,padb,padl,padr',padt,padb,padl,padr)
        print('ibeg,iend,jbeg,jend',ibeg,iend,jbeg,jend)

    imgtile = fill_value*np.ones([tdim[0],tdim[1],nb],dtype=img.dtype)
    imgtile[padt:padb,padl:padr] = np.atleast_3d(img[ibeg:iend,jbeg:jend])
    if transpose is not None:
        imgtile = imgtile.transpose(transpose)
    return imgtile

from scipy.sparse import csr_matrix

def compute_M(data):
    cols = np.arange(data.size)
    return csr_matrix((cols, (data.ravel(), cols)),
                      shape=(data.max() + 1, data.size))

def where_sparse(data):
    M = compute_M(data)
    return [np.unravel_index(row.data, data.shape) for row in M]

def where_numpy(data):
    return np.where(data)

def rotxy(x,y,adeg,xc,yc):
    """
    rotxy(x,y,adeg,xc,yc)

    Summary: rotate point x,y about xc,yc by adeg degrees

    Arguments:
    - x: x coord to rotate
    - y: y coord to rotate
    - adeg: angle of rotation in degrees
    - xc: center x coord
    - yc: center y coord

    Output:
    rotated x,y point
    """
    arad = DEG2RAD*adeg
    sinr,cosr = np.sin(arad),np.cos(arad)
    rotm = [[cosr,-sinr],[sinr,cosr]]
    xyp = np.dot(rotm,np.c_[x-xc,y-yc].T)

    # handle scalar outputs
    if xyp.ndim==2 and xyp.shape[1]==1:
        xyp = xyp.squeeze()
    return xyp[0]+xc,xyp[1]+yc
        
def meshpositions(*arrs):
    grid = np.meshgrid(*arrs)
    positions = np.vstack(list(map(np.ravel, grid)))
    return positions

def chull(p,**kwargs):
    from scipy.spatial import ConvexHull as _chull
    return_indices = kwargs.pop('return_indices',False)
    hullidx = _chull(p,**kwargs).vertices
    hullp = p[hullidx]
    if return_indices:
        return hullp,hullidx
    return hullp

def utm2latlon(easting,northing,zone,hemi='North',alpha=None,datum=DATUM_WGS84):
    if hemi not in ('North','South'):
        print('invalid hemisphere value=',hemi)
        return None,None
    
    zone_alpha = alpha or ('N' if hemi=='North' else 'M')
    lat,lon = UTMtoLL(datum,easting,northing,str(zone)+zone_alpha)
    return lat,lon

def sl2xy(s,l,**kwargs):
    """
    sl2xy(s,l,x0=0,y0=0,xps=0,yps=xps,rot=0,mapinfo=None) 

    Given integer pixel coordinates (s,l) convert to map coordinates (x,y)

    Arguments:
    - s,l: sample, line indices

    Keyword Arguments:
    - x0,y0: upper left map coordinate (default = (0,0))    
    - xps: x map pixel size (default=None)
    - yps: y map pixel size (default=xps)
    - rot: map rotation in degrees (default=0)
    - mapinfo: envi map info dict (entries replaced with kwargs above)    

    Returns:
    - x,y: x,y map coordinates of sample s, line l
    """
    mapinfo = kwargs.pop('mapinfo',{})    
    x0 = kwargs.pop('ulx',mapinfo.get('ulx',None))
    y0 = kwargs.pop('uly',mapinfo.get('uly',None))
    xps = kwargs.pop('xps',mapinfo.get('xps',None))
    yps = kwargs.pop('yps',mapinfo.get('yps',xps))
    rot = kwargs.pop('rot',mapinfo.get('rotation',0))

    if None in (x0,y0):
        raise ValueError("ulx or uly undefined")

    if None in (xps,yps):
        raise ValueError("xps or yps undefined")

    if yps == 0:
        yps = xps

    xp,yp = x0+xps*s, y0-yps*l        
    if rot == 0:
        return xp,yp

    ar = DEG2RAD*rot
    X, Y = rotxy(xp,yp,rot,x0,y0)
    #X = x0 + xps * s + ar  * l
    #Y = y0 + ar  * s - yps * l    
    return X, Y

def sl2latlon(s,l,**kwargs):
    mapinfo = kwargs.get('mapinfo',{})
    proj = mapinfo.get('proj',None)

    if not proj:
        raise ValueError("proj undefined")
    elif proj not in ('UTM','Geographic Lat/Lon'):   
        print('unknown projection:',proj)
        return None
    
    x,y = sl2xy(s,l,**kwargs)
    if proj=='Geographic Lat/Lon':
        return y,x
    elif proj=='UTM':
        return utm2latlon(y,x,zone=mapinfo['zone'],hemi=mapinfo['hemi'])
    else:
        raise Exception('Unknown projection "%s"'%proj)

def xy2sl(x,y,**kwargs):
    """
    xy2sl(x,y,x0=0,y0=,xps=0,yps=xps,rot=0,mapinfo=None) 

    Given a orthocorrected image find the (s,l) values for a given (x,y)

    Arguments:
    - x,y: map coordinates

    Keyword Arguments:
    - x0,y0: upper left map coordinate (default = (0,0))
    - xps: x map pixel size (default=None)
    - yps: y map pixel size (default=xps)
    - rot: map rotation in degrees (default=0)
    - mapinfo: envi map info dict (xps,yps,rot override)

    Returns:
    - s,l: sample, line coordinates of x,y
    """
    mapinfo = kwargs.pop('mapinfo',{})

    x0 = kwargs.pop('ulx',mapinfo.get('ulx',None))
    y0 = kwargs.pop('uly',mapinfo.get('uly',None))
    xps = kwargs.pop('xps',mapinfo.get('xps',None))
    yps = kwargs.pop('yps',mapinfo.get('yps',xps))
    rot = kwargs.pop('rot',mapinfo.get('rotation',0))
    #if mapinfo and rot != 0:
    #    # flip sign of mapinfo rot unless otherwise specified
    #    rot = rot * kwargs.pop('rotsign',-1) 
    
    if None in (x0,y0):
        raise ValueError("either ulx or uly defined")

    if xps is None:
        raise ValueError("pixel size defined")

    yps = yps or xps
    xp, yp = (x-x0), (y0-y)
    if rot!=0:
        xp, yp = rotxy(xp,yp,rot,0,0)

    xp,yp = xp/xps,yp/yps
    return xp,yp

def latlon2utm(lat,lon,zone=None,datum=DATUM_WGS84):
    """
    latlon2utm(lat,lon,zone=None,datum=DATUM_WGS84) 

    Arguments:
    - lat: latitude coordinate(s)
    - lon: longitude coordinate(s)

    Keyword Arguments:
    - zone:   UTM zone number (default=None, computed automatically)
    - datum:  reference ellipsoid (default=_DATUM_WGS84)

    Returns:
    - easting:  UTM easting map coordinate
    - northing: UTM northing map coordinate
    - zone:     UTM zone digit
    - hemi:     hemisphere letter (hemi >= 'N' -> Northern hemisphere)

    """

    zonealpha,easting,northing = LLtoUTM(datum,lat,lon,ZoneNumber=zone)
    return easting, northing, int(zonealpha[:-1]), zonealpha[-1]

def latlon2sl(lat,lon,**kwargs):
    mapinfo = kwargs.get('mapinfo',{})
    proj = mapinfo.get('proj',None)
    if not proj:
        raise ValueError("proj undefined")
    elif proj not in ('UTM','utm','Geographic Lat/Lon'):
        print('Unknown projection:',proj)
        return None
    
    if proj=='Geographic Lat/Lon':
        return xy2sl(lon,lat,mapinfo=mapinfo)

    zone = int(mapinfo['zone']) if 'zone' in mapinfo else None
    x,y,zone,zonealpha = latlon2utm(lat,lon,zone=zone)
    return xy2sl(x,y,mapinfo=mapinfo)

def latlon2xy(lat,lon,**kwargs):
    mapinfo = kwargs.get('mapinfo',{})
    proj = mapinfo.get('proj',None)
    if not proj:
        raise ValueError("proj undefined")
    elif proj not in ('UTM','Geographic Lat/Lon'):   
        print('unknown projection:',proj)
        return None
    
    if proj=='Geographic Lat/Lon':
        return lon,lat

    zone = int(mapinfo['zone']) if 'zone' in mapinfo else None
    x,y,zone,zonealpha = latlon2utm(lat,lon,zone=zone)    
    return x,y

def mapdict2str(mapdict):
    mapmeta = mapdict.pop('metadata',[])
    mapkeys,mapvals = list(mapdict.keys()),list(mapdict.values())
    nargs = 10 if mapdict['proj'].upper() =='UTM' else 7
    maplist = list(map(str,mapvals[:nargs]))
    mapkw = list(zip(mapkeys[nargs:],mapvals[nargs:]))
    mapkw = [str(k)+'='+str(v) for k,v in mapkw]
    mapstr = '{ '+(', '.join(maplist+mapkw+mapmeta))+' }'
    return mapstr

def mapinfo(img,astype=dict):
    from spectral import SpyFile
    _img = img if isinstance(img,SpyFile) else openimg(img)
    maplist = _img.metadata.get('map info',None)    
    if maplist is None or astype==list:
        return maplist    

    mapinfo = OrderedDict()
    mapinfo['proj'] = maplist[0]
    mapinfo['xtie'] = float(maplist[1])
    mapinfo['ytie'] = float(maplist[2])
    mapinfo['ulx']  = float(maplist[3])
    mapinfo['uly']  = float(maplist[4])
    mapinfo['xps']  = float(maplist[5])
    mapinfo['yps']  = float(maplist[6])

    if mapinfo['proj'] == 'UTM':
        mapinfo['zone']  = maplist[7]
        mapinfo['hemi']  = maplist[8]
        mapinfo['datum'] = maplist[9]

    mapmeta = []
    for mapitem in maplist[len(mapinfo):]:
        if '=' in mapitem:
            key,val = list(map(lambda s: s.strip(),mapitem.split('=')))
            mapinfo[key] = val
        else:
            mapmeta.append(mapitem)

    mapinfo['rotation'] = float(mapinfo.get('rotation','0'))
    if len(mapmeta)!=0:
        print('unparsed metadata:',mapmeta)
        mapinfo['metadata'] = mapmeta

    if astype==str:
        return mapdict2str(mapinfo)

    return mapinfo

def findhdr(img_file):
    from os import path
    dirname = path.dirname(img_file)
    filename,filext = path.splitext(img_file)
    if filext == '.hdr' and path.isfile(img_file): # img_file is a .hdr
        return img_file
    
    hdr_file = img_file+'.hdr' # [img_file.img].hdr or [img_file].hdr
    if path.isfile(hdr_file):
        return path.abspath(hdr_file)
    hdr_file = filename+'.hdr' # [img_file.img] -> [img_file].hdr 
    if path.isfile(hdr_file):
        return hdr_file
    return None

def openimg(imgf,hdrf=None,**kwargs):
    from spectral.io.envi import open as _open
    from spectral import SpyFile
    if isinstance(imgf,SpyFile):
        return imgf
    hdrf = hdrf or findhdr(imgf)
    return _open(hdrf,imgf,**kwargs)

def openmm(img,interleave='source',writable=False):
    if isinstance(img,str):
        _img = openimg(img)
        return _img.open_memmap(interleave=interleave, writable=writable)
    return img.open_memmap(interleave=interleave, writable=writable)

def openimgmm(imgf,interleave='source',writable=False):
    """
    openimgandmm(imgf,hdr=None,interleave='source',writable=False) 
    
    Arguments:
    - imgf: image file
    
    Keyword Arguments:
    - hdrf: hdr file corresponding to imgf (default=None)
    - interleave: image interleave (default='source')
    - writable: allow writing to memmap (default=False)
    
    Returns:
    - img: spectralpython img structure
    - img_mm: numpy memmap of img data
    """
    img = openimg(imgf)
    imgmm = openmm(img,interleave=interleave,writable=writable)
    return img,imgmm

def prob2geotiff(proboutf,probimg,probimgf):
    from gdal import Open, GetDriverByName, GA_ReadOnly, GDT_Float32
    
    print('saving',proboutf)  
    gtifdrv = GetDriverByName('Gtiff')
    nl_prob,ns_prob,nb_prob = probimg.shape
    g = Open(probimgf, GA_ReadOnly)
    geo_t = g.GetGeoTransform()
    geo_p = g.GetProjectionRef()
    rows,cols = g.RasterYSize,g.RasterXSize

    prob_geo_t = (1, geo_t[1], geo_t[2], 1, geo_t[4], geo_t[5])
    gsub = gtifdrv.Create(proboutf, ns_prob, nl_prob, nb_prob, GDT_Float32)
    gsub.SetGeoTransform(prob_geo_t)
    gsub.SetProjection(geo_p)
    for i in range(nb_prob):
        gsub.GetRasterBand(i+1).WriteArray(probimg[:,:,i])
    gsub = None

def gdalversion():
    return gdal.VersionInfo()
    
def gdalsl2xy(samp,line,geo_t):
    # maps sample,line coord to map x,y via GDAL affine geotransform matrix
    x,y = gdal.ApplyGeoTransform(geo_t,samp,line)
    #x = geo_t[0]+samp*geo_t[1]+line*geo_t[2]
    #y = geo_t[3]+samp*geo_t[4]+line*geo_t[5]
    return x,y

def gdalxy2sl(x,y,inv_t):
    # maps x,y coord to samp,line via GDAL affine geotransform matrix
    samp,line = gdal.ApplyGeoTransform(inv_t,x,y)
    return samp,line

def gdaldtype(a):
    
    gdtype = gdal_array.NumericTypeCodeToGDALTypeCode(a.dtype)
    if gdtype is None:
        raise Exception('No GDAL data type for numpy dtype: '+str(a.dtype))

    return gdtype

def gdalopen(imgfile,writable=False):
    mode = gdal.GA_ReadOnly if not writable else gdal.GA_Update
    return gdal.Open(imgfile,mode)

def proj2srs(geo_p):
    import osr    
    srs = osr.SpatialReference()
    srs.ImportFromWkt(geo_p)
    return srs

def gdalinfo(imgfile):
    import osr
    assert(gdalversion()[0] != '1')
    _gimg = gdalopen(imgfile)
    nlines = _gimg.RasterYSize
    nsamples = _gimg.RasterXSize
    nbands = _gimg.RasterCount
    dtype = _gimg.GetRasterBand(1).DataType
    meta = _gimg.GetMetadata()
    bandnames = [_gimg.GetRasterBand(bi+1).GetDescription()
                 for bi in range(nbands)]

    # populate projection fields
    geo_p = _gimg.GetProjectionRef()
    srs = proj2srs(geo_p)
    proj4 = srs.ExportToProj4()
    # convert srs proj info to ENVI "map info" format
    if '+proj=longlat' in proj4:
        proj = 'Geographic Lat/Lon'
    elif '+proj=utm' in proj4:
        proj = 'UTM'
    else:        
        proj = proj4

    # get image bounds + pixel size + rotation
    geo_t = _gimg.GetGeoTransform()
    inv_t = gdal.InvGeoTransform(geo_t)
    if inv_t is None:
        errmsg = 'Inverse geotransform failed (gdalversion=%s)'%str(gdalversion())
        raise RuntimeError(errmsg)
    xps = np.sqrt(geo_t[1]**2 + geo_t[4]**2)
    yps = np.sqrt(geo_t[2]**2 + geo_t[5]**2)
    ulx,uly = geo_t[0],geo_t[3]
    lrx,lry = gdalsl2xy(nsamples,nlines,geo_t)

    # if image is north up, sample=1,line=0 will be 0 deg from ulx,uly
    _xo,_yo = gdalsl2xy(1,0,geo_t)
    rotation = np.round(np.degrees(np.arctan2(_yo-uly, _xo-ulx)))

    del _xo,_yo,_gimg
    return locals()

def gdalwrite(outf,imgdata,imgmeta,geo_t,geo_p,nodata_value=None,
              nodata_mask=[],bandnames=[],outfmt='ENVI'):
    img = np.atleast_3d(imgdata)
    nl,ns,nb = img.shape
    if nodata_value is None:
        udtypes = (np.int8,np.uint8,np.uint16,np.uint32)
        nodata_value = 0 if imgdata.dtype in udtypes else -9999

    if len(bandnames)==0:
        bandnames = ['b%d'%bi for bi in range(nb)]

    gdtype = gdaldtype(img)
    outdrv = gdal.GetDriverByName(outfmt)        
    outimg = outdrv.Create(outf,ns,nl,nb,gdtype)
    if outimg is None:
        print('Unable to create outimg "%s"'%outf)
        raw_input()
        return
    outimg.SetGeoTransform(tuple(geo_t))
    outimg.SetProjection(geo_p)
    for bi in range(nb):
        outband = outimg.GetRasterBand(bi+1)
        # set band names        
        outband.SetDescription(bandnames[bi])

        # process + write img data
        banddata = img[:,:,bi]
        if len(nodata_mask)!=0:
            banddata = banddata.copy()
            banddata[nodata_mask] = nodata_value
        outband.WriteArray(banddata)

    if outfmt=='ENVI':
        imgmeta.setdefault('data ignore value',str(nodata_value))
    elif outfmt=='Gtiff':
        # set nodata on the last gtif band here (nodata shared by all bands) 
        outband.SetNoDataValue(nodata_value)
        
    outimg.SetMetadata(imgmeta,outfmt)
    outimg = None # flush to disk    

def geobbox(lat,lon,xydiam,inmap):
    # computes squere bounding box of size xydiam (meters) in utm, geo and pixel coords
    bbox_sl,bbox_xy,bbox_ll = np.zeros([3,4,2])
    # first convert to UTM in meters
    utmx,utmy,utmzone,utmalpha = latlon2utm(lat,lon)
    hemi = 'North' if lat>=0 else 'South'
    rotation,_xyoff = inmap.get('rotation',0), xydiam/2.0
    _xoff = [-_xyoff,  _xyoff, -_xyoff,  _xyoff]
    _yoff = [-_xyoff, -_xyoff,  _xyoff,  _xyoff]
    for i,(xo,yo) in enumerate(zip(_xoff,_yoff)):
        bbox_xy[i] = rotxy(utmx+xo,utmy+yo,rotation,utmx,utmy)
        bbox_ll[i] = utm2latlon(bbox_xy[i,1],bbox_xy[i,0],zone=utmzone,hemi=hemi)
        bbox_sl[i] = latlon2sl(bbox_ll[i,0],bbox_ll[i,1],mapinfo=inmap)
    del _xoff,_yoff,_xyoff
    return locals()

def tile2geotiff(tileimg,tilepos,tilef,baseimgf,validate_coords=True,
                 border='bbox'):
    import pylab as pl
    from gdal import Open, GetDriverByName, GA_ReadOnly, GDT_Byte

    print('saving',tilef)  
    gtifdrv = GetDriverByName('Gtiff')
    nl_tile,ns_tile,nb_tile = tileimg.shape
    g = Open(baseimgf, GA_ReadOnly)
    geo_t = g.GetGeoTransform()
    geo_p = g.GetProjectionRef()
    nl_img,ns_img = g.RasterYSize,g.RasterXSize

    if border == 'bbox':
        bordermask = np.ones([nl_tile,ns_tile],dtype=np.bool8)
    elif border=='circle':
        trad,tbufh = tileimg.shape[0]//2, 1
        bordermask = np.bool8(disk(trad+tbufh)[tbufh:-tbufh,tbufh:-tbufh])
    assert(bordermask.shape==tileimg.shape[:2])

    def sl2map(s,l,ulx,uly,xps):
        return sl2xy(s,l,mapinfo=dict(ulx=ulx,uly=uly,xps=xps,yps=xps,rot=0))

    def ct(s,l,mapdict):
        ulx,uly,xps = [mapdict[k] for k in ('ulx','uly','xps')]
        rot = mapdict.get('rotation',0)
        x,y = sl2map(s,l,ulx,uly,xps)
        if rot==0:
            return x,y
        return rotxy(x,y,rot,ulx,uly)

    def bbox(xy):
        minx,maxx = extrema([x for x,y in xy])
        miny,maxy = extrema([y for x,y in xy])
        return minx,maxy,maxx,miny

    mapdict = mapinfo(baseimgf)
    tile_ct = lambda sl: ct(sl[0],sl[1],mapdict)
    
    l0,s0 = tilepos
    tile_bbox_s = [s+s0 for s in [0,     0, ns_tile, ns_tile]]
    tile_bbox_l = [l+l0 for l in [0, nl_tile, 0,     nl_tile]]
    tile_bbox_sl = list(zip(tile_bbox_s,tile_bbox_l))
    tile_bbox_xy = list(map(tile_ct,tile_bbox_sl))
    
    #l0 = max(0,min(tilepos[0],nl_img-1))
    #s0 = max(0,min(tilepos[1],ns_img-1))
    #l1 = max(l0,min(tilepos[0]+tileimg.shape[0],ns_img-1))
    #s1 = max(s0,min(tilepos[1]+tileimg.shape[1],ns_img-1))
    x_tile = geo_t[0]+s0*geo_t[1]+l0*geo_t[2]
    y_tile = geo_t[3]+s0*geo_t[4]+l0*geo_t[5]
        
    if validate_coords:
        from skimage.measure import points_in_poly
        img_bbox_s = [0,0,ns_img,ns_img]
        img_bbox_l = [0,nl_img,nl_img,0]
        img_bbox_sl = list(zip(img_bbox_s,img_bbox_l))
        img_bbox_xy = []
        for s,l in img_bbox_sl:
            x = geo_t[0]+s*geo_t[1]+l*geo_t[2]
            y = geo_t[3]+s*geo_t[4]+l*geo_t[5]
            img_bbox_xy.append((x,y))

        off_lab = ['Upper Left','Lower Left','Upper Right','Lower Right']
        in_img = np.zeros(len(tile_bbox_sl),dtype=np.bool8)
        for i,(st,lt) in enumerate(tile_bbox_sl):
            xt = geo_t[0]+st*geo_t[1]+lt*geo_t[2]
            yt = geo_t[3]+st*geo_t[4]+lt*geo_t[5]
            xydist = (np.float32((xt,yt))-np.float32(tile_bbox_xy[i]))**2
            xydist = np.sqrt(xydist.sum())
            if xydist > 1.0:
                # if gdal/numpy differ by more than 1m, wait
                print('WARNING:','xydist=',xydist,'gdal (x,y)=',(xt,yt),
                      'numpy (x,y)=',tile_bbox_xy[i])
                raw_input()
            in_img[i] = points_in_poly([(xt,yt)],img_bbox_xy)[0]

        if not in_img.all():
            nzo = (in_img==0).sum()
            if nzo==4:
                print('WARNING: entire tile outside of image bounding box')
            else:
                print('WARNING: tile overlaps image bounding box (%d points outside)'%nzo)

            print(pathsplit(baseimgf)[1],'(s,l) -> (x,y) bounding box')
            for s,l in img_bbox_sl:
                print((s,l),'->\t',(x,y))        

            print('tile (s,l) -> (x,y) bounding box')
            for i,(st,lt) in enumerate(tile_bbox_sl):                
                print(off_lab[i],(st,lt),'->\t',(xt,yt),'in image bbox=',in_img[i])
                
            
    tile_geo_t = (x_tile, geo_t[1], geo_t[2], y_tile, geo_t[4], geo_t[5])
    gsub = gtifdrv.Create(tilef, ns_tile, nl_tile, nb_tile, GDT_Byte)
    gsub.SetGeoTransform(tile_geo_t)
    gsub.SetProjection(geo_p)
    for i in range(nb_tile):
        tileband = tileimg[:,:,i].copy()
        if border!='bbox':
            tileband[~bordermask] = 0
            # if i==0:
            #     import pylab as pl
            #     fig0,ax0 = pl.subplots(1,2,sharex=True,sharey=True)
            #     pl.suptitle(tilef)
            #     ax0[0].imshow(tileimg); ax0[0].set_xlabel('contour')
            #     ax0[1].imshow(bordermask); ax0[1].set_xlabel('boundary')
            #     pl.show()
        gsub.GetRasterBand(i+1).WriteArray(tileband)
    gsub = None # flush to disk

def rdn2rgb(rdn,rdnmin=0,rdnmax=15):
    #_rdnmin,_rdnmax = extrema(rdn,p=0.99)
    return np.clip((rdn-rdnmin)/(rdnmax-rdnmin),0.0,1.0)

def array2img(outf,img,mapinfostr=None,bandnames=None,**kwargs):
    outhdrf = outf+'.hdr'
    if pathexists(outf) and not kwargs.pop('overwrite',False):
        print('Cannot write image to path',outf)
        return
        
    img = np.atleast_3d(img)
    outmeta = dict(samples=img.shape[1], lines=img.shape[0], bands=img.shape[2],
                   interleave='bip')
    
    outmeta['file type'] = 'ENVI'
    outmeta['byte order'] = 0
    outmeta['header offset'] = 0
    outmeta['data type'] = envitypecode(img.dtype)

    if mapinfostr:
        mapinfostr = mapinfostr.strip()
        assert (mapinfostr[0] == '{') and (mapinfostr[-1] == '}')
        outmeta['map info'] = mapinfostr

    if bandnames:
        outmeta['band names'] = '{%s}'%", ".join(bandnames)
        
    outmeta.setdefault('data ignore value',NODATA)

    outimg = createimg(outhdrf,outmeta)
    outmm = openmm(outimg,writable=True)
    outmm[:] = img
    outmm = None # flush output to disk
    print('saved %s array to %s'%(str(img.shape),outf))

def mad(a,axis=0,medval=None,unbiased=False):    
    '''
    computes the median absolute deviation of a list of values
    mad = median(abs(a - medval))/c
    '''
    from statsmodels.robust.scale import mad as _mad
    center = medval or np.median(a,axis=axis)
    c = 0.67448975019608171 if unbiased else 1.0
    # return np.median(np.abs(np.asarray(a)-medval))
    return _mad(a, c=c, axis=axis, center=center)
    
def kde(img,k):
    from scipy.ndimage import gaussian_filter
    imgkde = gaussian_filter(img,sigma=k,truncate=1)
    imgkde = (imgkde-imgkde.min())/(imgkde.max()-imgkde.min())
    return img*imgkde

def absnorm(img,mask):
    assert((len(img.shape)==2))
    print('normalizing image to absolute range')
    i32 = np.float32(img)
    imax = np.abs(i32[~mask]).max()
    imin = -imax
    imgn=np.clip((i32-imin)/(imax-imin),0.0,1.0)
    return imgn,imin,imax

def smoothbil(img, mask, d, sigmaColor, sigmaSpace, normalize=True):
    from cv2 import bilateralFilter
    print('running bilateralFilter')
    if normalize:
        imgn,imin,imax  = absnorm(img,mask)
    else:
        imgn = img.copy()
        imin,imax = extrema(img[~mask])
    imgn = bilateralFilter(imgn, d, sigmaColor, sigmaSpace)
    imgn = imin+(imgn*(imax-imin))
    return imgn

def relabel_sequential(labimg,**kwargs):
    from skimage.segmentation import relabel_sequential as _rls
    return _rls(labimg,**kwargs)

def remove_small_objects(labimg,min_size, **kwargs):
    from skimage.morphology import remove_small_objects as _rso
    #set values if not passed in **kwargs
    #kwargs.setdefault('min_size',9)
    kwargs.setdefault('in_place',False)
    #print('filtering objects smaller than %.2f' % kwargs['min_size'])
    return _rso(labimg, min_size, **kwargs)

def filtdet(ch4mf,nodata_mask,mfmapinfo,minarea=minarea,mfmin=mfmin,mfmax=mfmax,
            k=kernel,mfminsmall=mfminsmall,skip_kde=False,use_abs=False,
            kde_outf=None,ccomp_outf=None,det_outf=None):
    from skimage.morphology import reconstruction
    print('filtering weakly-connected detections below minppm=%.2fppmm'%mfmin)
    mfmapstr=mapdict2str(mfmapinfo)
    detkde = np.abs(ch4mf) if use_abs else ch4mf.copy()
    ch4min = ch4mf >= mfmin
    if not skip_kde:
        detkde = kde(detkde,k=k)
    detkde = np.clip((detkde-mfmin)/(mfmax-mfmin),0.0,1.0)

    if not skip_kde and kde_outf:
        array2img(kde_outf,detkde,mapinfostr=mfmapstr,overwrite=True)
    detmask = (detkde>0) 
    print('%d candidate components'%imlabel(detmask).max())
    if 0:
        print('%d components before hole removal'%imlabel(detmask).max())    
        seed = detmask.copy()
        seed[1:-1, 1:-1] = detmask.max()
        print('removing interior holes')        
        detmask = np.bool8(reconstruction(seed, detmask, method='erosion'))
        print(extrema(ch4mf[detmask]))
        print('%d components after hole removal'%imlabel(detmask).max())

    print('removing detections with <= minarea=%d pixels'%(minarea))
    detsmall = detmask.copy()
    detmask = remove_small_objects(detmask,min_size=minarea,in_place=False)
    print('%d remaining detections'%imlabel(detmask).max())
    if mfminsmall >= mfmin:
        print('adding small detections with mf>=%f ppmm'%(mfminsmall))
        smallcc = imlabel(detsmall!=detmask)
        smallkeep = np.unique(smallcc[(ch4mf>=mfminsmall)])
        smallkeep = smallkeep[smallkeep!=0]
        print(len(smallkeep),'small detections found')
        smallmask = (np.in1d(smallcc.ravel(),smallkeep).reshape(detmask.shape))
        detmask = (detmask | np.bool8(smallmask)) 

    # exclude nodata+ch4min afterward to exclude interior holes from ccimg
    detcomp = imlabel(detmask)
    detcomp[~ch4min] = 0
    detcomp,_,_ = relabel_sequential(detcomp)
    print('%d final detections'%detcomp.max())
    
    if ccomp_outf:
        detcomp[nodata_mask] = NODATA
        array2img(ccomp_outf,detcomp,mapinfostr=mfmapstr,overwrite=True)

    detkde[~ch4min] = 0
    detkde[nodata_mask] = 0
    detcomp[nodata_mask] = 0
    
    print('detected %d components'%detcomp.max())
    if det_outf:
        detfilt = ch4mf.copy()
        detfilt[~ch4min]=0        

        detfilt[nodata_mask] = NODATA
        array2img(det_outf,detfilt,mapinfostr=mfmapstr,overwrite=True)

    return detkde, detcomp

def loadmaskedimage(maskedimgf,rgb_bands=[],masked_value=np.nan,
                    load_bands=[],memmap=False,verbose=0):
    maskeddir,maskedfile = pathsplit(maskedimgf)    
    maskedimg = envi_open_file(maskedimgf.split('.')[0] +'.hdr',image=maskedimgf)
    rows,cols,bands = maskedimg.shape
    if verbose:
        print('Loading [%d,%d,%d] masked input image: "%s"'%(rows,cols,bands,maskedimgf))
    if memmap:
        maskeddata = maskedimg.open_memmap(interleave='source',writeable=False)
    else:
        nload=len(load_bands)
        if nload>1:
            maskeddata = maskedimg.read_bands(load_bands)
        elif nload==1:
            maskeddata = maskedimg.read_bands([load_bands[0]])
        else:
            maskeddata = maskedimg.load() # load everything

    maskeddata = np.float32(maskeddata)
    if maskeddata.ndim == 2:
        maskeddata = maskeddata[...,np.newaxis]
    nodata_value = float(maskedimg.metadata.get('data ignore value',np.nan))
    nodata_mask = (maskeddata==nodata_value).any(axis=2)
    if not memmap:
        maskeddata[nodata_mask] = masked_value
    else:
        masked_value = nodata_value

    #calculate number of no data pixels
    n_nodata_pixels = np.count_nonzero(nodata_mask)
    if verbose or n_nodata_pixels>0:
        print(np.count_nonzero(nodata_mask),'nodata pixels in image',maskedimgf)

    outdata = dict(mapinfo=mapinfo(maskedimg,astype=dict),
                   nodata_mask=nodata_mask,
                   nodata_value=nodata_value)

    if bands>=3 and len(rgb_bands)==3:
        # ang cmf image: 3 rgb + 1 cmf band
        image_bands = list(set(range(bands))-set(rgb_bands))
        outdata['rgb'] = rdn2rgb(maskeddata[:,:,rgb_bands])
        if len(image_bands)!=0:
            outdata['image'] = maskeddata[:,:,image_bands].squeeze()
    elif bands==2 and len(set(rgb_bands))==1:
        # avcl cmf image: 1 cmf band + 1 grayscale 
        image_bands = list(set(range(bands))-set(rgb_bands))
        outdata['rgb'] = rdn2rgb(maskeddata[:,:,rgb_bands])
        if len(image_bands)!=0:
            outdata['image'] = maskeddata[:,:,image_bands].squeeze()        
    else:
        outdata['image'] = maskeddata.squeeze()
    
    return outdata

def rgb2labimg(rgbimg):
    assert(rgbimg.shape[2]==3)
    
    labimg = np.zeros(rgbimg.shape[:2],dtype=np.uint8)

    # point sources = [255,0,0], diffuse sources = [0,0,255]
    # false source = [0,255,255]
    rgbsum = rgbimg.sum(axis=2)
    posmask = rgbsum==(1*255)
    labimg[posmask & (rgbimg[:,:,0]==255)] = POINTSRC
    labimg[posmask & (rgbimg[:,:,2]==255)] = DIFFSRC
    labimg[~posmask & (rgbimg[:,:,1:]==255).all(axis=2)] = FALSESRC

    # source location = [255,255,255]
    labimg[rgbsum==(3*255)] = LOCSRC
    
    return labimg

def labimg2rgb(labimg,rgba=False):
    rows,cols = labimg.shape[:2]
    rgbimg = np.zeros([rows,cols,3],dtype=np.uint8)

    rgbimg[labimg==POINTSRC,0] = 255
    rgbimg[labimg==DIFFSRC, 2] = 255
    rgbimg[labimg==FALSESRC,1:] = 255

    if rgba:
        islab = np.isin(labimg,(POINTSRC,DIFFSRC,FALSESRC))
        rgbimg = np.dstack([rgbimg,255*islab])
        
    return rgbimg


def loadlabimg(labf):
    print('loading label mask image: "%s"'%labf)
    labpath,labfile = pathsplit(labf)
    filebase,fileext = splitext(labfile)
    if fileext=='.png':
        labimg = imread(labf)
        if labimg.shape[2] in (3,4):
            labimg = rgb2labimg(labimg[:,:,:3]).squeeze()

    elif fileext=='' and filebase.endswith('class'):
        # envi class map
        imgdat = envi_open_file(labf+'.hdr',image=labf)
        #labimg = imgdat.open_memmap(interleave='source',writeable=False).copy()
        labimg = imgdat.load().squeeze()
    else:
        raise Exception('Unrecognized format %s'%labf)

    labimg = np.uint8(labimg)
    assert np.isin(np.unique(labimg),CMFLABELS).all()
    
    return labimg

def loadfiltdet(detfilt_imgf):
    print('loading filtered detection image: "%s"'%detfilt_imgf)    
    detdir,detfile = pathsplit(detfilt_imgf)    
    detimg = envi_open_file(detfilt_imgf+'.hdr',image=detfilt_imgf)
    ch4det = np.float32(detimg.load().squeeze())
    nodata_value = float(detimg.metadata['data ignore value'])
    nodata_mask = ch4det==nodata_value
    ch4det[nodata_mask] = 0
    return dict(ch4det=ch4det,mapinfo=mapinfo(detimg,astype=dict),
                nodata_mask=nodata_mask,nodata_value=nodata_value)

def loaddetids(detid_imgf):
    print('loading detection id image: "%s"'%detid_imgf)    
    detdir,detfile = pathsplit(detid_imgf)    
    detimg = envi_open_file(detid_imgf+'.hdr',image=detid_imgf)
    detids = np.float32(detimg.load().squeeze())
    detmeta = detimg.metadata
    nodata_value = float(detmeta['data ignore value'])
    nodata_mask = detids==nodata_value
    detids[nodata_mask] = 0
    return dict(detids=detids,mapinfo=mapinfo(detimg,astype=dict),
                nodata_mask=nodata_mask,nodata_value=nodata_value)

def loadsaliencemap(salience_imgf):
    print('loading',salience_imgf)
    saliencedir,saliencefile = pathsplit(salience_imgf)
    salienceimg = envi_open_file(salience_imgf+'.hdr',image=salience_imgf)
    saliencemapinfo = mapinfo(salienceimg,astype=dict)
    saliencemap = np.float32(salienceimg.load().squeeze())
    return dict(saliencemap=saliencemap,mapinfo=saliencemapinfo)

def loadcmf(filepath,rdnmin=0,rdnmax=15):
    img,mm = openimgmm(filepath,writable=False)
    imgmap = mapinfo(img)
    nodata_value = float(img.metadata.get('data ignore value'))
    dat = mm[:,:,:].copy()
    assert(dat.shape[2]==4)
    cmf = np.float32(dat[...,3])
    nodata = cmf==nodata_value
    rgb = np.float32(dat[...,:3])
    rgb = np.clip((rgb-rdnmin)/(rdnmax-rdnmin),0.0,1.0)
    rgb = np.dstack([rgb,np.float32(nodata==0)])
    return cmf,rgb,nodata,imgmap

def plotcmf(cmfdat,minppmm=250,maxppmm=1500,minrdn=0,maxrdn=15,nodata=[]):
    assert(cmfdat.ndim==3 and cmfdat.shape[-1]==4)
    cmf = cmfdat[...,-1]
    cmfmask = cmf<minppmm
    rgb = np.clip((cmfdat[...,:-1]-minrdn)/(maxrdn-minrdn),0.0,1.0)
    if len(nodata)!=0:
        cmfmask |= nodata
        rgb = np.dstack([rgb,np.float32(nodata==0)])
    cmf = np.ma.masked_where(cmfmask, cmf)
    fig,ax = pl.subplots(1,1)
    ax.imshow(rgb,vmin=0,vmax=1,alpha=0.5)
    ax.imshow(cmf,vmin=minppmm,vmax=maxppmm,cmap='YlOrRd')
    return fig,ax

def bbox(points,border=0,imgshape=[]):
    """
    bbox(points) 
    computes bounding box of extrema in points array

    Arguments:
    - points: [N x 2] array of [rows, cols]
    """
    from numpy import atleast_2d
    points = atleast_2d(points)
    minv = points.min(axis=0)
    maxv = points.max(axis=0)
    difv = maxv-minv
    
    if isinstance(border,list):
        rborder,cborder = border
        rborder = rborder if rborder > 1 else int(rborder*difv[0])
        cborder = cborder if cborder > 1 else int(cborder*difv[1])
    elif border < 1:
        rborder = cborder = int(border*difv.mean()+0.5)
    elif border == 'mindiff':
        rborder = cborder = min(difv)
    elif border == 'maxdiff':
        rborder = cborder = max(difv)
    else:
        rborder = cborder = border
        
    if len(imgshape)==0:
        imgshape = maxv+max(rborder,cborder)+1
    
    rmin,rmax = max(0,minv[0]-rborder),min(imgshape[0],maxv[0]+rborder+1)
    cmin,cmax = max(0,minv[1]-cborder),min(imgshape[1],maxv[1]+cborder+1)    
    
    return (rmin,cmin),(rmax,cmax)
    
def maskbbox(mask,border=0):
    """
    maskbbox(mask,border=0) 
    computes the bounding box of nonzero pixel coords for input mask
    
    Arguments:
    - mask: [n x m] binary image mask
    
    Keyword Arguments:
    - border: number of pixels to use in padding bounding box (default=0)
    
    Returns:
    - bbox = (rowmin,colmin),(rowmax,colmax)
    """
    
    points = np.c_[np.where(mask)]
    return bbox(points,border,mask.shape)

def region_maxima(img,mask,return_index=False):
    from skimage.measure import regionprops
    ccimg = imlabel(mask)
    ulab = np.unique(ccimg[ccimg!=0])
    rcidx,rcmax = [],[]
    for r in regionprops(ccimg,intensity_image=img):        
        rcmax.append(r.max_intensity)
        if return_index:
            rc=r.coords
            rcidx.append(r.coords[np.argmax(img[rc[:,0],rc[:,1]])])
    rcmax=np.array(rcmax,dtype=img.dtype)
    if return_index:
        return rcmax,np.array(rcidx,dtype=np.int)
    return rcmax

def local_maxima(im,rad,image_max=[]):
    from skimage.feature import peak_local_max
    
    diam = 2*rad
    if 0:
        # image_max is the dilation of im with a diam x diam structuring element
        # It is used within peak_local_max function
        from scipy.ndimage import maximum_filter
        if len(image_max)==0:
            image_max = np.zeros(im.shape[:2])
        maximum_filter(im, size=diam, mode='constant', output=image_max)
    
    # Comparison between image_max and im to find the coordinates of local maxima
    return peak_local_max(im, min_distance=diam)

def formatcmd(cmd_tpl,cmd_args):
    return ' '.join(cmd_tpl.format(**cmd_args).split('\n')).strip()

def savecmdlogs(log_prefix,log_stdout,log_stderr,encoding='utf-8'):
    str_stdout = str(log_stdout.decode(encoding))
    str_stderr = str(log_stderr.decode(encoding))
    log_outf = log_prefix+'.out'
    log_errf = log_prefix+'.err'
    print('log_stdout:\n%s'%log_outf)
    print('log_stderr:\n%s'%log_errf)
    with open(log_outf,'w') as out_fid:
        out_fid.write(str_stdout)
    with open(log_errf,'w') as err_fid:
        err_fid.write(str_stderr)

def runcmd(cmd,logprefix=None,verbose=0):
    from subprocess import Popen, PIPE
    cmdstr = ' '.join(cmd) if isinstance(cmd,list) else cmd
    if verbose:
        print("running command:",cmdstr)
    cmdout = PIPE
    for rstr in ['>>','>&','>']:
        if rstr in cmdstr:
            cmdstr,cmdout = list(map(lambda s:s.strip(),cmdstr.split(rstr)))
            mode = 'w' if rstr!='>>' else 'a'
            cmdout = open(cmdout,mode)
            
    p = Popen(cmdstr.split(), stdout=cmdout, stderr=cmdout)
    out, err = p.communicate()
    retcode = p.returncode

    if cmdout != PIPE:
        cmdout.close()

    if verbose:
        print('command completed with return code "%d"'%retcode)

    if logprefix is not None:
        savecmdlogs(logprefix,out,err)
    
    return out,err,retcode

def bounds2rgba(bndimg,color=(0.,1.,0.,1.),bgcolor=(0.,0.,0.,0.),
                astype=np.uint8, **kwargs):
    """
    bounds2rgba(bndimg)
    
    Summary: converts a boundary image to an alpha masked rgba image
    
    Arguments:
    - bndimg: boundary image
    
    Keyword Arguments:
    None
    
    Output:
    - rgba boundary image
    """
    rows,cols = bndimg.shape[:2]
    rgba = np.zeros([rows,cols,4])
    rgba[bndimg!=0] = color
    if bgcolor != (0.,0.,0.,0.):
        rgba[bndimg==0] = bgcolor

    if astype==np.uint8:
        rgba = rgba*255
        
    return astype(rgba)

def retrieve_rgb(rgbf):
    wgetbin='wget --no-verbose'
    wgeturl='https://avirisng.jpl.nasa.gov'
    wgetrgb='{wgetbin} -O {rgbdir}/{rgbfile} {wgeturl}/aviris_locator/y{rgbyear}_RGB/{rgbfile}'
    wgetgeo='{wgetbin} -O {rgbdir}/{rgbfile} {wgeturl}/ql/{rgbyear}qlook/{lid}_geo.jpeg'

    wgetretc = 1
    lid = filename2flightid(rgbf)
    if not lid.startswith('ang'):
        raise Exception('retrieve_rgb only works with AVIRIS-NG flightlines')

    if pathexists(rgbf):        
        return 0
    
    try:
        (rgbdir,rgbfile),rgbyear = pathsplit(rgbf),lid[5:7]
        print(rgbdir,rgbfile,rgbyear)
        wgetcmd = wgetrgb if rgbyear!='17' else wgetgeo
        wgetcmd = wgetcmd.format(**vars())
        print(wgetcmd)
        wgetout = runcmd(wgetcmd)
        wgetretc = wgetout[-1]
        if wgetretc != 0:
            print(rgbf,'wget failure, skipping')
            print('wget output:', wgetout[1])
        
    except Exception as e:
        print(rgbf,'not found and unable to retreive, skipping')        

    return wgetretc

def string_ticklabels(ticklabels):
    import pylab as pl
    if isinstance(ticklabels[0],pl.Text):
        return [l.get_text() for l in ticklabels]
    return [str(l) for l in labels]

def shift_ticklabels(ticklabels,shift):
    import pylab as pl
    fmtfn = lambda s: str(float(s)+shift)
    if isinstance(ticklabels[0],pl.Text):
        ticklabels = string_ticklabels(ticklabels)
    return modify_ticklabels(ticklabels,fmt=fmtfn)

def scale_ticklabels(ticklabels,scale):    
    fmtfn = lambda s: str(float(s)*scale)
    return modify_ticklabels(ticklabels,fmt=fmtfn)
  
def modify_ticklabels(ticklabels,fmt='%0.2f'):
    import pylab as pl
    fmtfn = fmt if not isinstance(fmt,str) else lambda s: fmt%s
    labels = ticklabels
    if isinstance(labels[0],pl.Text):
        labels = string_ticklabels(labels)
        
    return [fmtfn(l) for l in labels]

counts = lambda a: OrderedDict(zip(*np.unique(a,return_counts=True)))

def gcdist(dlon1, dlat1, dlon2, dlat2):
    """
    gcdist(dlon1, dlat1, dlon2, dlat2)

    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)

    Input:
    - dlon1,dlat1: first lat/lon coordinate
    - dlon2,dlat2: second lat/lon coordinate

    Returns:
    - distance in meters between (dlon1,dlat1) and (dlon2,dlat2)
    """
    # convert decimal degrees to radians haversine formula
    lon1,lat1,lon2,lat2 = [np.radians(c) for c in [dlon1,dlat1,dlon2,dlat2]]
    a = np.sin((lat2-lat1)/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2-lon1)/2)**2
    return 12742000 * np.arcsin(np.sqrt(a)) # meters 6371*1000*2
    
def parse_masks(lid,cid,lat,lon,maskdir,masksuf,maskwin=3):
    """
    parse_masks(lid,lon,lat,maskdir,masksuf,maskwin=3,dfcols=dfcols)

    Summary: parse artifact masks for a set of candidates 

    Arguments:
    - cid: candidate ids (nx1 string)
    - lid: flightline ids (nx1 string)
    - lon: longitude (nx1 float)
    - lat: latitude (nx1 float)
    - maskdir: maskdir
    - masksuf: masksuf

    Keyword Arguments:
    - maskwin: maskwin (default=3)

    Output:
    - output
    """
    from pandas import DataFrame
    dfcols = ['lid','cid','lat','lon','row','col']
    assert((len(lid)==len(lon)) and (len(lat)==len(lon)))
    # maskwin must be an odd number >= 3
    assert((maskwin >= 3) & (maskwin % 2 == 1))
    maskrad = maskwin//2
    maskcols = []
    dfout = []
    ulid = np.unique(lid)
    step = max(1,len(ulid)//10)
    for i,ilid in enumerate(ulid):
        if i%step==0:
            print('processing flightline artifact mask %d of %d'%(i+1,len(ulid)))
        maskpath = pathjoin(maskdir,ilid+'*'+masksuf)
        print('maskpath: "%s"'%str((maskpath)))
        maskf = glob(maskpath)
        if len(maskf)==0:
            print('mask for lid "%s" not found'%str((ilid)))
            raw_input()
            continue
        elif len(maskf)!=1:
            print('multiple masks for lid "%s" found (using first):'%ilid,maskf)
            raw_input()

        maskf = maskf[0]
        maskimg = openimg(maskf)
        
        nodatav = maskimg.metadata.get('data ignore value',-9999)
        maskmap = mapinfo(maskimg)
        maskbands = maskimg.metadata['band names']
        if len(maskcols)==0:
            #maskcols = [bn.split()[0]+'_mask' for bn in maskbands]
            maskcols = [bn.split()[0] for bn in maskbands]
        assert(len(maskcols)==maskimg.shape[2])
        lididx = np.where(lid==ilid)[0]
        nline, nsamp = maskimg.shape[0], maskimg.shape[1]

        for idx in lididx:
            icid,ilon,ilat = cid[idx],lon[idx],lat[idx]
            s,l = latlon2sl(ilat,ilon,mapinfo=maskmap)
            # search a maskwin x maskwin window centered on [l,s]
            lmin = int(max(0,min(round(l)-maskrad,nline-1)))
            lmax = int(min(l+maskrad+2,nline-1))
            smin = int(max(0,min(round(s)-maskrad,nsamp-1)))
            smax = int(min(s+maskrad+2,nsamp-1))
            maskroi = maskimg[lmin:lmax,smin:smax]            
            maskroi = (maskroi!=0) & (maskroi!=nodatav)
            maskroi = np.uint8(maskroi.reshape(-1,len(maskcols)).any(axis=0))        
            dfout.append([ilid,icid,ilat,ilon,l,s]+list(maskroi))

        del maskimg # flush memory map

    maskdf = DataFrame(dfout,columns=dfcols+maskcols)
    return maskdf,maskcols

def mask2rgb(imgmask,alpha=1.0):
    maskbands = ['cloud','specular','flare','dark']
    maskcolor = dict(cloud=(0.8, 0.8, 0.8, alpha),
                     specular=(0.8, 0.6, 0.2, alpha),
                     flare=(0.9, 0.1, 0.0, alpha),
                     dark=(0.0, 0.0, 0.0, alpha))
    maskrgb = np.zeros(imgmask.shape)
    nodata = (imgmask==-9999).all(axis=2)
    for i,bandi in enumerate(maskbands):
        maski = (imgmask[:,:,i]!=0) & ~nodata
        maskrgb[maski] = maskcolor[bandi]
    return maskrgb

    
def labpng2tif(pngf,labf,meta,overwrite=False):
    if os.path.exists(labf) and not overwrite:
        print(f'{labf} exists')
        return True
    labimg = imread(pngf).transpose(2,0,1)
    return labimg2tif(labimg,labf,meta)

def labimg2tif(labimg,labf,meta):
    print('labimg.shape: "%s"'%str((labimg.shape)))
    assert((labimg.shape[1]==meta['height']) &
           (labimg.shape[2]==meta['width']))
    meta['count'] = labimg.shape[0]
    meta['dtype'] = 'uint8'
    meta['driver'] = 'GTiff'
    meta.pop('nodata',None)
    with rio.open(labf, 'w', **meta) as dst:
        dst.write(labimg)
    return os.path.exists(labf)

def ime_scale(ps):
    #           ppm(m)      ps^2        L/m^3       mole/L      kg/mole
    scalef = (1.0/1e6)*((ps*ps)/1.0)*(1000.0/1.0)*(1.0/22.4)*(0.01604/1.0)
    return scalef

def ime(pixels_ppmm,ps):
    assert (np.isfinite(pixels_ppmm) & (pixels_ppmm>=0)).all()
    return pixels_ppmm.sum()*ime_scale(ps)
        
def bbox_overlap(bb1,bb2,pixel_coords=True):
    assert bb1['xmin'] < bb1['xmax']
    assert bb1['ymin'] < bb1['ymax']
    assert bb2['xmin'] < bb2['xmax']
    assert bb2['ymin'] < bb2['ymax']
    
    # determine the coordinates of the intersection rectangle
    x_l = max(bb1['xmin'], bb2['xmin'])
    x_r = min(bb1['xmax'], bb2['xmax'])
    y_b = min(bb1['ymax'], bb2['ymax'])
    y_t = max(bb1['ymin'], bb2['ymin'])

    if x_r < x_l or y_b < y_t:
        return 0

    pixel_inc = 1 if pixel_coords else 0
    area_overlap = (x_r-x_l+pixel_inc) * (y_b-y_t+pixel_inc)
    return area_overlap
    
def iou(bb1,bb2,**kwargs):
    assert bb1['xmin'] < bb1['xmax']
    assert bb1['ymin'] < bb1['ymax']
    assert bb2['xmin'] < bb2['xmax']
    assert bb2['ymin'] < bb2['ymax']
    
    area_overlap = bbox_overlap(bb1,bb2,**kwargs)
    if area_overlap==0:
        return 0.0

    # compute the area of both AABBs
    area_bb1 = bb1['width'] * bb1['height']
    area_bb2 = bb2['width'] * bb2['height']

    iou = area_overlap / float(area_bb1+area_bb2-area_overlap)

    assert iou >= 0.0
    assert iou <= 1.0

    if kwargs.get('verbose',False):
        print(f'area_bb1={area_bb1}, area_bb2={area_bb2}, '+
              f'area_overlap={area_overlap}, iou={iou}')
    
    return iou


if __name__ == '__main__':
    import pylab as pl
    cmff= 'ang20200924t211102_ch4mf_v2y1_img'
    cmf,rgb,nodata,cmfmap = loadcmf(cmff)
    plotcmf(np.dstack([rgb[...,:-1]*15,cmf]),maxrdn=15,nodata=nodata)
    pl.show()
    img = np.random.rand(500,500)
    ul = (-10,-20)
    tdim = (100,100)
    t=extract_tile(img,ul,tdim,verbose=1).squeeze()
    figkw = dict(figsize=(2*7.5,1*7.5+0.25),sharex=True,sharey=True)
    fig,ax = pl.subplots(1,2,**figkw)
    ax[0].imshow(img)
    ax[1].imshow(t)
    pl.tight_layout()
    pl.show()
    
    labf = 'ang20200709t205211_ch4mf_v2y1_img_mask.png'
    labimg = loadlabimg(labf)
    figkw = dict(figsize=(1*7.5,1*7.5+0.25),sharex=True,sharey=True)
    fig,ax = pl.subplots(1,1,**figkw)
    ax.imshow(labimg)
    pl.tight_layout()
    pl.show()

    inmap=gdalinfo('/Users/bbue/GSuite/My Drive/Eddy Detection /S1AB_GeoTiff/S1A_IW_GRDH_1SDV_20181208T142437_20181208T142502_024935_02BF4E_5668.tif')
    gbox=geobbox(36.56,-124.48,20000,inmap)
    print('geobbox(124.48,36.56,20000,inmap): "%s"'%str((gbox)))
    sl = gbox['bbox_sl']
    print('sl[:,0].max()-sl[:,0].min(): "%s"'%str((sl[:,0].max()-sl[:,0].min())))
    print('sl[:,1].max()-sl[:,1].min(): "%s"'%str((sl[:,1].max()-sl[:,1].min())))
    raw_input()
    
    imgdir = '/Users/bbue/Research/srcfinder/permian/'
    imgfiles = ['ang20190927t182905_msk_v2x1_img',
                'ang20190922t190407_msk_v2x1_img',
                'ang20191013t152730_msk_v2x1_img']
    for imgf in imgfiles:
        imgf = pathjoin(imgdir,imgf)
        print('imgf: "%s"'%str((imgf)))
        imgmap = mapinfo(imgf)
        print('mapinfo[rotation]: "%s"'%str((imgmap['rotation'])))
        geomap = gdalinfo(imgf)
        print('geoinfo[rotation]: "%s"'%str((geomap['rotation'])))
    raw_input()    
    rgbf= 'tiles/tag_mini_intensive/rgb/ang20160922t183143_RGB.jpeg'
    print(retrieve_rgb(rgbf))
    raw_input()
    cmap = 'inferno'
    tilefloat = np.random.rand(100,100)
    tilergba = float2rgba(tilefloat,cmap=cmap)
    rgbafloat = rgba2float(tilergba,cmap=cmap)
    diff = np.sqrt((tilefloat-rgbafloat)**2)
    print(rgbafloat.shape,diff.mean())
    import pylab as pl
    fig,ax = pl.subplots(1,4,sharex=True,sharey=True)
    ax[0].imshow(tilefloat,cmap=cmap,vmin=0,vmax=1)
    ax[1].imshow(tilergba[...,:3])
    ax[2].imshow(rgbafloat,cmap=cmap,vmin=0,vmax=1)
    ax[3].imshow(diff,vmin=0,vmax=0.01)
    pl.show()
    
    if 0:
        baseimg='./data/y16/cmf/ort/ang20160913t205656_cmf_v1n2_img'
        #baseimg='/lustre/ang/y16/cmf/ort/ang20160913t205656_cmf_v1n2_img'    
        tileimg = 255*np.ones([100,100,4],dtype=np.uint8)
        tilepos=(12360,728)
        tilefile='test.tif'
        tile2geotiff(tileimg,tilepos,tilefile,baseimg)
        raw_input()
        z=np.random.rand(10)
        z[3:6] = np.nan
        y=np.random.rand(10,10)
        y[1:5] = np.nan
        r = array2rgba(z)
        print(r,r.shape)
        r = array2rgba(y)
        print(r,r.shape)
        raw_input()
    import pylab as pl
    datadir='./data'    
    #datadir='/lustre/ang'
    if 1:
        cmfdir=datadir+'/y16/cmf/ort'
        labdir=pathjoin(cmfdir,'thorpe_training')
        detdir='tiles/thorpe_training/100'

        labimgf=pathjoin(labdir,'ang20160914t180328_cmf_v1n2_img_class')
        cmfimgf=pathjoin(cmfdir,'ang20160914t180328_cmf_v1n2_img')
        #filtdetimgf=pathjoin(detdir,'ang20160914t180328_cmf_v1n2_filt_det_500_1500')
        filtdetimgf='./tiles/thorpe_training/150/ang20160914t180328_cmf_v1n2_filt_det_350_1500'
        
    else:
        cmfdir=datadir+'/y15/cmf/ort'
        labdir=pathjoin(cmfdir,'thompson_training')
        detdir='tiles/thompson_training/100/'
        labimgf=pathjoin(labdir,'ang20150419t161445_cmf_v1f_img_mask.png')
        cmfimgf=pathjoin(cmfdir,'ang20150419t161445_cmf_v1f_img')        
        filtdetimgf = pathjoin(detdir,'ang20150419t161445_cmf_v1f_filt_det_500_1500')

    tiledim=150
    tilepos = 4678,1115
    tileimgf = './tiles/thorpe_training/150/ang20160914t180328/tp/tp_det4678_1115.png'
    pngimgf=filtdetimgf+'.png'
    sys.path.insert(0,'/Users/bbue/Research/src/python/util/tilepredictor/')
    from tilepredictor_util import imread_rgb,imread_tile
    
    tileimg = imread_tile(tileimgf,tile_shape=[tiledim,tiledim])
    pngimg = imread_rgb(pngimgf)
    tilepng = pngimg[tilepos[0]:tilepos[0]+tiledim,tilepos[1]:tilepos[1]+tiledim]
    cmfdata = loadmaskedimage(cmfimgf,rgb_bands=range(3))        
    detdata = loadfiltdet(filtdetimgf)    
    labimg = loadlabimg(labimgf)
    tilelab = labimg[tilepos[0]:tilepos[0]+tiledim,tilepos[1]:tilepos[1]+tiledim]
    # make sure nodata masks match 
    nodata_mask = detdata['nodata_mask']
    assert(cmfdata['nodata_mask'].sum()==nodata_mask.sum())

    ch4det = detdata['ch4det']    
    rgbimg = cmfdata['rgb']
    # remove labeled pixels in nodata regions
    labimg = np.float32(labimg)
    ch4det[nodata_mask | (ch4det<500)] = np.nan
    labimg[~(labimg>0)] = np.nan


    
    fig,ax = pl.subplots(3,1,sharex=True,sharey=True,num=1)
    ax[0].imshow(tileimg)
    ax[1].imshow(tilepng)
    ax[2].imshow(tilelab)
    pl.show()

    pl.clf()
    ax[0].imshow(rgbimg.swapaxes(0,1))
    ax[1].imshow(pngimg.swapaxes(0,1))
    pl.show()
    
    ax[0].imshow(rgbimg.swapaxes(0,1))
    ax[0].imshow(ch4det.swapaxes(0,1),vmin=500,vmax=1500,cmap='YlOrRd')
    ax[1].imshow(labimg.swapaxes(0,1),vmin=0,vmax=1,cmap='Spectral',alpha=0.7)
    ax[1].imshow(ch4det.swapaxes(0,1),vmin=500,vmax=1500,cmap='YlOrRd')
    pl.show()

