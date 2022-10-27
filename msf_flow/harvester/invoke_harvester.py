from datetime import datetime
import boto3
import json

def invoke_harvester(date):
	"""
	logic to launch 24 parallel lambda harvesters (1 per hour) so as to avoid 
	timeout and take advantage of parallelism
	Args:
		date (str): date to harvest data in format YYYY-MM-DDThh:mm:ssZ
	"""
	date_fmt = "%Y-%m-%d"
	date_fmt_precise = "%Y-%m-%dT%H:%M:%SZ"
	date = datetime.strptime(date,date_fmt_precise).strftime(date_fmt)

	# initialize boto3 client for aws lambda
	client = boto3.client('lambda')
	req = {}
	data = {}

	# download 4 files at a time (every 4 hours)
	for i in range(0,24,4):
		data['start_date'] = '{date}T{hr:02d}:00:00Z'.format(date=date,hr=i)
		data['end_date'] = '{date}T{hr:02d}:59:59Z'.format(date=date,hr=i+3)

		# invoke harvest lambda on hrrr_hourly_utah for entire day
		data['dataset'] = 'hrrr_hourly_utah'
		print()
		print(data)
		resp = client.invoke(
			FunctionName='harvest-rtma',
			InvocationType='Event',
			LogType='Tail',
			Payload=bytes(json.dumps(data),encoding='utf8')
		)
		print(resp)

		# invoke harvest lambda on hrrr_hourly_nomads_filtered for entire day
		data['dataset'] = 'hrrr_hourly_nomads_filtered'
		print()
		print(data)
		resp = client.invoke(
			FunctionName='harvest-rtma',
			InvocationType='Event',
			LogType='Tail',
			Payload=bytes(json.dumps(data),encoding='utf8')
		)
		print(resp)
	
	# download 4 files at a time (every hour)
	# invoke harvest lambda over each hour of the day for each dataset
	for i in range(24):
		data['start_date'] = '{date}T{hr:02d}:00:00Z'.format(date=date,hr=i)
		data['end_date'] = '{date}T{hr:02d}:59:59Z'.format(date=date,hr=i)


		# invoke harvest lambda on rtma_15min_noaa
		data['dataset'] = 'rtma_15min_noaa'
		print()
		print(data)
		resp = client.invoke(
			FunctionName='harvest-rtma',
			InvocationType='Event',
			LogType='Tail',
			Payload=bytes(json.dumps(data),encoding='utf8')
		)
		print(resp)


def lambda_handler(event,context):
	invoke_harvester(date=event['time'])
	return {
        'statusCode': 200,
        'body': json.dumps('Done launching harvesters!')
    }