import argparse
import json
import pygrib
import smtplib
import subprocess
from datetime import datetime
from botocore.exceptions import ClientError
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication
from utils.logger import init_logger

SENDER = ""
RECIPIENTS = '' # multiple emails separated by spaces


def parse_args():
    """Retrieve command line parameters.
    
    Returns:
        ArgumentParse: command line parameters
    """
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("-f", "--filename", required=True,
                        help="file to check")
    args = parser.parse_args()
    return args


def send_email_aws(email_body="",subject="",logger=None):
    import boto3

    CHARSET = "utf-8"
    SUBJECT = subject
    AWS_REGION = "us-west-2"
    BODY_TEXT = email_body

    # Create a new SES resource and specify a region.
    client = boto3.client('ses',region_name=AWS_REGION)

    # Create a multipart/mixed parent container.
    msg = MIMEMultipart('mixed')
    # Add subject, from and to lines.
    msg['Subject'] = SUBJECT 
    msg['From'] = SENDER 
    msg['To'] = ','.join(RECIPIENTS.split(' '))

    # Create a multipart/alternative child container.
    msg_body = MIMEMultipart('alternative')

    # Encode the text and HTML content and set the character encoding
    # necessary if sending message with characters outside ASCII range
    textpart = MIMEText(BODY_TEXT.encode(CHARSET), 'plain', CHARSET)
    
    # Add the text and HTML parts to the child container.
    msg_body.attach(textpart)

    # Attach the multipart/alternative child container to the multipart/mixed parent container.
    msg.attach(msg_body)

    try:
        #Provide the contents of the email.
        response = client.send_raw_email(
            Source=SENDER,
            Destinations=RECIPIENTS,
            RawMessage={
                'Data':msg.as_string(),
            }
        )

    # Display an error if something goes wrong. 
    except ClientError as e:
        logger.error(e.response['Error']['Message'])
    else:
        logger.warning("Email sent! Message ID:")
        logger.warning(response['MessageId'])

def send_email_local(email_body="",file="",logger=None):
    cmd = "echo {} | mailx -s \"{} Data Harvest Error $(date -u '+%D')\" {}".format(email_body,file,RECIPIENTS)

    o = subprocess.check_output(cmd,shell=True)
    if not logger is None:
        logger.warning(o)


def main(filename=None,bucket_name="bucket",aws=False):
    # f = '/Users/eyam/M2AF/data/wind/rtma_15min_noaa/20191006/rtma2p5_ru.201910062345z.2dvaranl_ndfd.grib2'
    logger = init_logger()
    f = filename
    contents = []
    failed = False
    email_body = ''
    now = datetime.now().strftime('%Y/%m/%dT%H:%M:%S')

    # if file from s3, download to tmp and set f to that path
    if aws:
        f = '/tmp/'+filename.split('/')[-1]
        import boto3
        s3 = boto3.resource("s3")
        target_bucket = s3.Bucket(bucket_name)
        target_bucket.download_file(filename,f)

    try:
        grbs = pygrib.open(f)
        for grb in grbs:
            if not logger is None:
                logger.info(grb)
            contents.append(grb)
        print(contents)

        if len(contents) > 0:
            if not logger is None:
                logger.warning('success')
        else:
            if not logger is None:
                logger.warning('file empty')
            failed = True
            email_body = 'File {} was downloaded but found to be empty.'.format(f)


    except:
        if not logger is None:
            logger.error('file could not be read')
        failed = True
        email_body = 'File {} download attempted but could not be read.'.format(f)

    finally:
        if failed:
            subject = "Data Harvest Error {} {}".format(f,now)
            if aws:
                send_email_aws(email_body,subject,logger)
            else:
                send_email_local(email_body,subject,logger)
        else:
            if not logger is None:
                logger.info('Checked harvested wind data file '+f)


def lambda_handler(event, context):
    bucket = event['Records'][0]["s3"]["bucket"]["name"]
    filename = event['Records'][0]["s3"]["object"]["key"]
    main(filename,bucket,True)
    return {
        'statusCode': 200,
        'body': json.dumps('Wind Data Quality Check Lambda completed successfully.')
    }


if __name__ == '__main__':
    args = parse_args()
    print(args)
    main(args.filename)
