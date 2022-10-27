import csv
import json
import os, sys
import numpy as np
import scipy as sp
import scipy.sparse as sparse
import scipy.sparse.linalg
import pandas as pd
import matplotlib.pyplot as plt
import boto3

def run_inversion(bucket_name='bucket', rsel3_key=None, AWS=False, \
    grid_dir=None, trop_mat2_local=None, rsel3_local=None):
    if AWS:
        # download inputs
        prefix = '/'.join(rsel3_key.split('/')[:-1])
        if 'example_inversion_input' in rsel3_key:
            prefix = prefix.replace('example_inversion_output', 'example_inversion_input')

        else:
            prefix = prefix.replace('inversion_outputs', 'model_inputs')

        lat_key = prefix+'/lat.csv'
        lon_key = prefix+'/lon.csv'
        trop_mat2_key = prefix+'/trop_H.csv'
        # lat_key = 'data/example_inversion_input/sjvFebApr2020/lat.csv'
        # lon_key = 'data/example_inversion_input/sjvFebApr2020/lon.csv'
        # trop_mat2_key = 'data/example_inversion_input/sjvFebApr2020/trop_H.csv'
        # rsel3_key = 'data/example_inversion_output/sjvFebApr2020/trop_meta.csv'

        # bucket_name = 'bucket'
        bucket = boto3.resource('s3').Bucket(bucket_name)

        lat_local = '/tmp/'+lat_key
        lon_local = '/tmp/'+lon_key
        trop_mat2_local = '/tmp/'+trop_mat2_key
        rsel3_local = '/tmp/'+rsel3_key

        rsel3_dir = '/'.join(rsel3_local.split('/')[:-1])
        if not os.path.isdir(rsel3_dir):
            os.makedirs(rsel3_dir)

        latlon_dir = '/'.join(lat_local.split('/')[:-1])
        if not os.path.isdir(latlon_dir):
            os.makedirs(latlon_dir)

        print('download {} to {}'.format(lat_key, lat_local))
        bucket.download_file(lat_key, lat_local)
        print('download {} to {}'.format(lon_key, lon_local))
        bucket.download_file(lon_key, lon_local)
        print('download {} to {}'.format(trop_mat2_key, trop_mat2_local))
        bucket.download_file(trop_mat2_key, trop_mat2_local)
        print('download {} to {}'.format(rsel3_key, rsel3_local))
        bucket.download_file(rsel3_key, rsel3_local)

    else:
        lat_local = os.path.join(grid_dir, 'lat.csv')
        lon_local = os.path.join(grid_dir, 'lon.csv')
        rsel3_dir = '/'.join(rsel3_local.split('/')[:-1])

    # inversion calculation
    rsel3 = pd.read_csv(rsel3_local).drop(labels='Unnamed: 0.1', axis=1)
    trop_mat2 = pd.read_csv(trop_mat2_local, header=None)

    y = rsel3.xch4 - rsel3.back
    H = trop_mat2.copy()

    x_a = np.array([10] * H.shape[1])
    Hx_a = np.dot(H, x_a)

    print('get spdiags')

    Sa = sparse.spdiags((x_a*.5) ** 2, diags=0, n=len(x_a), m=len(x_a))
    # Sa = np.diag(v = x_a*0.5)
    So = sparse.spdiags((y*.1) ** 2, diags=0, n=len(y), m=len(y))
    # So = np.diag(v = y*0.1)
    # print((So.todense()).shape)

    print('start inverse equations')

    #Inverse equations
    term1 = Sa.dot(H.T)
    print(term1.shape)
    term2 = H.dot(term1) + So.todense()
    term3 = sparse.linalg.spsolve(term2, y - Hx_a)
    # term3 = np.linalg.solve(term2, y - Hx_a)
    x_hat = x_a + term1.dot(term3)

    print('got x_hat')
    print(x_hat[0])

    term4 = sparse.linalg.spsolve(So, H)
    a = H.T.dot(term4)
    b = sparse.linalg.inv(Sa)
    S_hat = np.linalg.inv(a + b.todense())

    print('got s_hat')
    print(S_hat[0])

    x_hat_local = rsel3_dir+'/x_hat.csv'
    S_hat_local = rsel3_dir+'/S_hat.csv'

    np.savetxt(x_hat_local, x_hat, delimiter=",")
    print('x_hat written to csv {}'.format(x_hat_local))

    np.savetxt(S_hat_local, S_hat, delimiter=",")
    print('S_hat written to csv {}'.format(S_hat_local))

    if AWS:
        x_hat_key = x_hat_local.split('/tmp/')[1]
        S_hat_key = S_hat_local.split('/tmp/')[1]

        bucket.upload_file(x_hat_local, x_hat_key)
        print('{} uploaded to s3://{}/{}'.format(x_hat_local, bucket_name, x_hat_key))
        bucket.upload_file(S_hat_local, S_hat_key)
        print('{} uploaded to s3://{}/{}'.format(S_hat_local, bucket_name, S_hat_key))

def plot_result(grid_dir):
    #Plot result
    lat_local = os.path.join(grid_dir, 'lat.csv')
    lon_local = os.path.join(grid_dir, 'lon.csv')
    x_hat_local = os.path.join(grid_dir, 'x_hat.csv')

    lat = pd.read_csv(lat_local)
    lon = pd.read_csv(lon_local)
    x_hat = pd.read_csv(x_hat_local)
    print(x_hat)

    x_plot = np.reshape(x_hat, (len(lat), len(lon)))
    # x,y = np.meshgrid(lon, lat)

    # plt.pcolormesh(x, y, x_plot, vmin=0, vmax=np.percentile(x_plot, 95))
    # plt.colorbar()
    # plt.show()
    # plot_local = rsel3_dir+'/plot.png'
    # plt.savefig(plot_local)
    # plot_key = plot_local.split('/tmp/')[1]
    # bucket.upload_file(plot_local, plot_key)
    # print('{} uploaded to s3://{}/{}'.format(plot_local, bucket_name, plot_key))

def lambda_handler(event, context):
    bucket = event['Records'][0]["s3"]["bucket"]["name"]
    key = event['Records'][0]["s3"]["object"]["key"]
    run_inversion(bucket_name=bucket, rsel3_key=key, AWS=True)
    return {
        'statusCode': 200,
        'body': json.dumps('Hello from Lambda!')
    }

if __name__ == '__main__':
    grid_dir = sys.argv[1]
    # trop_mat2_path = sys.argv[2]
    # rsel3_path = sys.argv[3]
    # run_inversion(grid_dir=grid_dir, trop_mat2_local=trop_mat2_path, rsel3_local=rsel3_path)
    plot_result(grid_dir)
