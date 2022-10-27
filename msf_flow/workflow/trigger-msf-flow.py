import json
import boto3
from datetime import datetime

def lambda_handler(event, context):
    bucket = event['Records'][0]["s3"]["bucket"]["name"]
    key = event['Records'][0]["s3"]["object"]["key"]  # 20200909thh:mm:ss-plume-list.csv

    client = boto3.client('batch')

    now = datetime.strftime(datetime.utcnow(),"%Y%m%dT%H%M%SZ")
    job_name = 'msf-flow-{}'.format(now)
    job_queue = 'arn:aws:batch:us-west-2:510343621499:job-queue/m2af-job-queue'
    job_definition = 'msf-flow'
    parameters = {}
    parameters['PLUMEDIR'] = 's3://{bucket}/{key}'.format(bucket=bucket, key=key)
    parameters['WINDIR'] = 's3://bucket/data/wind/'
    parameters['OUTPATH'] = 's3://bucket/data/cmf/ch4/ort/plumes/ch4mfm_v2x1_img_detections/plumes_ext'

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
