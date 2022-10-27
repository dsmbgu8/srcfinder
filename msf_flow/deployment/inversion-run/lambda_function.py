import boto3
import json
import csv
import os
import numpy as np
import scipy as sp
import scipy.sparse as sparse
import scipy.sparse.linalg
import pandas as pd
import zipfile
from netCDF4 import Dataset
#import asyncio
#import matplotlib.pyplot as plt

# uses np-mpl-pd-gpd layer
# arn:aws:lambda:us-west-2:510343621499:layer:np-mpl-pd-gpd:1

# For this test, do basic inversion equations taken from Rodgers (2000)
#	Here, x_hat (solution) is solved analytically by minimizing and objective function
#	that weights model-data mismatch and prior knowledge about emissions
#
#	x_hat = x_a + G(y - Hx_a)
#
#		where x_a = prior
#			G = Gain matrix; G = (S_aH^T)(HSaH^T + So)^-1
#				where Sa = prior error covariance, So = observational error covariance
#			H = Jacobian (STILT footprints)
#			y = TROPOMI observations	
#
#	S_hat = (H^TSo^-1H + Sa^-1)^-1 is the posterior error covariance

# For this test, we assume error covariance matrices are diagonal, 
# and that So = 10% of y
# and that Sa = 50% of x_a
#
# AND we assume x_a is uniform

# These are bad assumptions, but will test if inversion computationally works. Can make more complicated later,

# You can do other fancy things like sum up emissions, error, etc, but it all basically comes down to
# operating on the x_hat and S_hat variables. So maybe these should get saved and exported
def create_nc_file_latlon2d(a, lat_vals, lon_vals, fname, varname="fluxes",
                            varunits=None, fill=None, logger=None):
    assert len(a.shape) == 2
    lat_dim, lon_dim = a.shape
    rootgrp = Dataset(fname, "w", format="NETCDF4")
    rootgrp.createDimension("lat", lat_dim)
    rootgrp.createDimension("lon", lon_dim)
    
    vals = rootgrp.createVariable("fluxes", "f4",
                                  dimensions=("lat", "lon",),
                                  fill_value=fill)
    
    lats = rootgrp.createVariable("lat", "f4", dimensions=("lat",))
    lons = rootgrp.createVariable("lon", "f4", dimensions=("lon",))
    vals[:, :] = a
    lats[:] = lat_vals
    lons[:] = lon_vals
    
    if varunits is not None:
        vals.units = varunits
    
    lats.units = "degrees north"
    lons.units = "degrees east"
    rootgrp.close()
    if logger is not None:
        logger.info("Wrote {}".format(fname))

def create_nc_file_timelatlon(a, time_vals, lat_vals, lon_vals, fname, varname=None,
              varunits=None, fill=None, logger=None):

  print(a.shape)
  #assert len(a.shape) == 3
  lat_dim, lon_dim = a.shape
  rootgrp = Dataset(fname, "w", format="NETCDF4")
  rootgrp.createDimension("time", time_dim)
  rootgrp.createDimension("lat", lat_dim)
  rootgrp.createDimension("lon", lon_dim)
  '''
  vals = rootgrp.createVariable(varname, "f4",
                 dimensions=("time","lat", "lon",),
                 fill_value=fill)
  time_var = rootgrp.createVariable("time", "f4", dimensions=("time",))
  lat_var = rootgrp.createVariable("lat", "f4", dimensions=("lat",))
  lon_var = rootgrp.createVariable("lon", "f4", dimensions=("lon",))
  vals[:, :] = a
  time_var[:] = time_vals
  lat_var[:] = lat_vals
  lon_var[:] = lon_vals
  if varunits is not None:
    vals.units = varunits
  time_var.units = "seconds since 1970-01-01 00:00:00"
  lat_var.units = "degrees north"
  lat_var.units = "degrees east"
  rootgrp.close()
  if logger is not None:
    logger.info("Wrote {}".format(fname))
  '''


def run_inversion(bucket_name, input_zip):
    # download inputs
    bucket = boto3.resource('s3').Bucket(bucket_name)
    input_zip_localp = '/tmp/' + input_zip.split('/')[-1]
    print('download source list file {} to {}'.format(input_zip,input_zip_localp))
    bucket.download_file(input_zip,input_zip_localp)
    input_zip = input_zip_localp
        
    
    zfile = zipfile.ZipFile(input_zip)
    dirname = None
    
    #get the input dir file name
    for x in zfile.namelist():
        if x.endswith('/'):
            dirname =x.split('/')[0] 
    print("dirname : {}".format(dirname))

    with zipfile.ZipFile(input_zip, 'r') as zip_ref:
        zip_ref.extractall('/tmp/')

    rootDir = "{}/{}".format('/tmp', dirname)
    for dirName, subdirList, fileList in os.walk(rootDir):
        print('Found directory: %s' % dirName)
        for fname in fileList:
            print('\t%s' % fname)

    prefix = rootDir
    lat_local = prefix+'/lat.csv'
    lon_local = prefix+'/lon.csv'
    trop_mat2_local = prefix+'/trop_H.csv.gz'
    rsel3_local  = prefix +'/trop_meta.csv'
    netcdf_file = prefix + "/result.nc"

    '''
    # inversion calculation
    rsel3 = pd.read_csv(rsel3_local).drop(labels='Unnamed: 0.1', axis=1)
   
    trop_mat2 = pd.read_csv(trop_mat2_local,header=None)
    print("{} read_csv done".format(trop_mat2_local))
    
    y = rsel3.xch4 - rsel3.back
    H = trop_mat2.copy()

    #print("H : \n{}".format(H))
    x_a = np.array([10] * H.shape[1])
    #print("(H, x_a) : {}, {}".format(H, x_a))
    Hx_a = np.dot(H, x_a)

    print('get spdiags')

    Sa = sparse.spdiags((x_a*.5) ** 2, diags=0, n=len(x_a), m=len(x_a))
    # Sa = np.diag(v = x_a*0.5)
    So = sparse.spdiags((y*.1) ** 2, diags=0, n=len(y), m=len(y))
    #print("So.todense().shape : {}".format((So.todense()).shape))

    print('start inverse equations')

    #Inverse equations
    try:
        term1 = Sa.dot(H.T)
        
        print("term1.shape : {}".format(term1.shape))
        term2 = H.dot(term1) + So.todense()
        print("term2.shape : {}".format(term2.shape))
        #term3 = sparse.linalg.spsolve(term2, y - Hx_a)
        term3 = np.linalg.solve(term2, y - Hx_a)

        print("term3.shape : {}".format(term3.shape))
    
        x_hat = x_a + term1.dot(term3)
    except Exception as err:
        print("Exception : {}".format(str(err)))
        raise Exception(str(err))

    print('got x_hat')
    print(np.mean(x_hat))
    print("mean done")
    print(x_hat[0])
    
    
    x_hat_mean = np.mean(x_hat)
    print("x_hat_mean : {}".format(x_hat_mean))
    
    
    #For Now we dont need S-hat
    term4 = sparse.linalg.spsolve(So, H)
    a = H.T.dot(term4)
    b = sparse.linalg.inv(Sa)
    S_hat = np.linalg.inv(a + b.todense())
    
    print('got s_hat')
    print(S_hat[0])
    '''

    #Load data
    H = pd.read_csv(trop_mat2_local, header=None)
    rsel3 = pd.read_csv(rsel3_local)
    y = rsel3.xch4 - 1860 #Set background to 1860
    lon = np.loadtxt(lon_local)
    lat = np.loadtxt(lat_local)
    
    #Do extremely simple inversion - ordinary least squares
    HTH = H.T.dot(H)
    x_hat = np.linalg.solve(HTH, H.T.dot(y))
 
    #rsel3_dir = '/'.join(rsel3_dir.split('/')[:-1])
    x_hat_local = prefix+'/x_hat.csv'
    #S_hat_local = rsel3_dir+'/S_hat.csv'

    # with open(x_hat_local, 'w+'):
        # x_hat.to_csv(path_or_buf=f, index=True)
    np.savetxt(x_hat_local, x_hat, delimiter=",")
    print('x_hat written to csv {}'.format(x_hat_local))

    # with open(S_hat_local, 'w+'):
        # S_hat.to_csv(path_or_buf=f, index=True)
    #np.savetxt(S_hat_local, S_hat, delimiter=",")
    #print('S_hat written to csv {}'.format(S_hat_local))

    #x_hat_key = x_hat_local.split('/tmp/')[1]
    #S_hat_key = S_hat_local.split('/tmp/')[1]
    output_path = 'data/inversion_run_test/output/{}/x_hat.csv'.format(dirname)
    bucket.upload_file(x_hat_local, output_path)
    print('{} uploaded to s3://{}/{}'.format(x_hat_local, bucket_name, output_path))
    #bucket.upload_file(S_hat_local, S_hat_key)
    #print('{} uploaded to s3://{}/{}'.format(S_hat_local, bucket_name, S_hat_key))


    #Regrid result
    x_plot = np.reshape(x_hat, (len(lat), len(lon)))
    print(x_plot)

    time_vals = None
    varname = None
 
    #create_nc_file_timelatlon(x_plot, time_vals, lat, lon, "result.nc", varname, None, None, None)
    create_nc_file_latlon2d(x_plot, lat, lon, netcdf_file, varname)
    output_path = 'data/inversion_run_test/output/{}/{}'.format(dirname, os.path.basename(netcdf_file))
    bucket.upload_file(netcdf_file, output_path)

def lambda_handler(event, context):
    print(event)
    bucket = event['Records'][0]["s3"]["bucket"]["name"]
    
    key = event['Records'][0]["s3"]["object"]["key"]
    run_inversion(bucket,key)
    return {
        'statusCode': 200,
        'body': json.dumps('Hello from Lambda!')
    }

if __name__ == '__main__':
    run_inversion()

