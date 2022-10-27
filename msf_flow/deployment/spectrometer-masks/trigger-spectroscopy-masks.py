import json
import boto3
from datetime import datetime

def lambda_handler(event, context):
    bucket_name = event['Records'][0]["s3"]["bucket"]["name"]
    # data/flightlines/fl.txt
    key = event['Records'][0]["s3"]["object"]["key"]
    
    bucket = boto3.resource('s3').Bucket(bucket_name)

    txt_path_local = '/tmp/'+key.split('/')[-1]
    print('download {} to {}'.format(key, txt_path_local))
    bucket.download_file(key, txt_path_local)

    files = []
    with open(txt_path_local, "r") as fd:
        files = fd.read().splitlines()

    client = boto3.client('batch')

    now = datetime.strftime(datetime.utcnow(), "%Y%m%dT%H%M%SZ")
    job_name = 'spectroscopy-masks-{}'.format(now)
    job_queue = 'arn:aws:batch:us-west-2:510343621499:job-queue/m2af-job-queue'
    job_definition = 'spectrometer-masks'
    parameters = {
        'BUCKET': bucket_name,
        'TXT': '\n'.join(files),
        'INPATH': 'data/rdn/ort/',
        'OUTPATH': 'data/masks/'
    }

    response = client.submit_job(
        jobDefinition=job_definition,
        jobName=job_name,
        jobQueue=job_queue,
        parameters=parameters
    )

    return {
        'statusCode': 200,
        'body': json.dumps(response)
    }
