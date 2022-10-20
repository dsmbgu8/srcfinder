#!/usr/bin/env python
# coding: utf-8

import os
import argparse
from glob import glob
import matplotlib.pyplot as plt
import rasterio as rio
import numpy as np
from astropy.convolution import convolve_fft
import subprocess

def gkern(l=5, sig=1.):
    """    creates gaussian kernel with side length `l` and a sigma of `sig`
    """
    ax = np.linspace(-(l - 1) / 2., (l - 1) / 2., l)
    gauss = np.exp(-0.5 * np.square(ax) / np.square(sig))
    kernel = np.outer(gauss, gauss)
    return kernel / np.sum(kernel)

def get_pixsig(ores, tres):
    # sigma in meters
    sig = tres / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    # sigma in pixels
    return sig / ores


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate gaussian downsampled flightlines")

    parser.add_argument('srcfl',                type=str, 
                                                help='Source flightline filename')

    parser.add_argument('dstfl',                type=str,
                                                help='Destination flightline filename')

    parser.add_argument('-res', '-r',           type=float,
                                                help='Target resolution',
                                                default=30)

    parser.add_argument('--preproc', '-p',      action='store_true',
                                                help='Preprocess to North-up')

    parser.add_argument('--nodata',             type=float,
                                                default=-9999,
                                                help='NODATA value of dstfl')

    args = parser.parse_args()

    if args.preproc:
        # If the flightline is rotated, convert to northup
        subprocess.run([
            'gdalwarp',
            '-of', 'GTiff',
            '-srcnodata', '-9999',
            '-dstnodata', '-9999',
            args.srcfl,
            'northup_fl.tif'
        ])
        srcfl = 'northup_fl.tif'
    else:
        srcfl = args.srcfl

    with rio.open(srcfl) as ds:
        # Read data
        data = ds.read(1)
        res = ds.res[0]
        
        # Set nodata to nan
        data[data==ds.nodata] = np.nan

        # Calculate sigma of Gaussian PDF
        sig = get_pixsig(res, args.res)
        # Derive kernel width, comes out to >3sig
        kerw = int(np.ceil((args.res * np.sqrt(2) * 2) / res))
        # Make sure kernel is odd width
        if kerw % 2 == 0: kerw += 1
        # Get Gaussian kernel
        ker = gkern(l=kerw, sig=sig)

        # FFT Convolve width astropy
        # We use astropy because of its superior nan handling
        # nans and boundaries are filled with 0, nans are kept as nans
        blurred = convolve_fft(
            data,
            ker,
            boundary='fill',
            fill_value=0,
            nan_treatment='fill',
            preserve_nan=True,
            allow_huge=True
        )
        # replace nans with NODATA value
        blurred[np.isnan(blurred)] = args.nodata

        new_profile = ds.profile
        new_profile['driver'] = 'GTiff'

        with rio.Env():
            with rio.open('temp_fl', 'w', **ds.profile) as ods:
                ods.write(blurred.astype(rio.float32), 1)

    subprocess.run([
        'gdal_translate',
        '-r', 'nearest',
        '-tr', '30', '30',
        #'-outsize', str(int((res/args.res)*ds.shape[1])), str(int((res/args.res)*ds.shape[0])),
        'temp_fl',
        args.dstfl
    ]) 

    for f in glob('northup_fl*'):
        os.remove(f)
    for f in glob('temp_fl*'):
        os.remove(f)