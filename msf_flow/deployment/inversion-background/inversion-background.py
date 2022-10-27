import json
import csv
import os
import numpy as np
import pandas as pd
import boto3

# uses np-mpl-pd-gpd layer
# arn:aws:lambda:us-west-2:510343621499:layer:np-mpl-pd-gpd:1

def lambda_handler(event, context):
    # download input files
    # rsel2_key = 'data/example_inversion_input/sjvFebApr2020/trop_meta.csv'
    # bucket_name = 'bucket'
    rsel2_key = event['Records'][0]["s3"]["object"]["key"]
    bucket_name = event['Records'][0]["s3"]["bucket"]["name"]

    bucket = boto3.resource('s3').Bucket(bucket_name)

    rsel2_local = '/tmp/'+rsel2_key
    rsel2_dir = '/'.join(rsel2_local.split('/')[:-2])
    if not os.path.isdir(rsel2_dir):
        os.makedirs(rsel2_dir)

    print('download {} to {}'.format(rsel2_key, rsel2_local))
    bucket.download_file(rsel2_key, rsel2_local)

    rsel2 = pd.read_csv(rsel2_local)

    rsel_back = rsel2[['xch4', 'posix']].groupby('posix').apply(lambda x: np.percentile(x, 5))
    rsel_back = rsel_back.reset_index()
    rsel_back.columns = ['posix', 'back']

    rsel3 = pd.merge(rsel2, rsel_back, on='posix', how='left')

    if 'example_inversion_input' in rsel2_key:
        out_key = rsel2_key.replace('example_inversion_input', 'example_inversion_output')

    else:
        out_key = rsel2_key.replace('model_inputs', 'inversion_outputs')

    # add "background subfolder to output path"
    fname = out_key.split('/')[-1]
    out_key = '/'.join(out_key.split('/')[:-1] + ['background']) + fname
    out_local = '/tmp/'+out_key

    out_dir = '/'.join(out_local.split('/')[:-1])
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    with open(out_local, 'w+') as f:
        rsel3.to_csv(path_or_buf=f, index=True)
        print('background dataframe written to csv {}'.format(out_local))

    bucket.upload_file(out_local, out_key)
    print('output csv {} uploaded to s3://{}/{}'.format(out_local, bucket_name, out_key))

    return {
        'statusCode': 200,
        'body': json.dumps('Hello from Lambda!')
    }
