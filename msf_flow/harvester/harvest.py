import sys, os
import argparse
from datetime import datetime, timezone, timedelta
import logging
import urllib.request
import yaml
import json

def parse_args():
    """Retrieve command line parameters.
    
    Returns:
        ArgumentParse: command line parameters
    """
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("-ds", "--dataset_name",
                        help="name of dataset to harvest")
    parser.add_argument("-b", "--data_basedir",
                        help="path to local dataset directory")
    parser.add_argument("-s", "--start_date",
                        help="start date to harvest (YYYYMMDD)")
    parser.add_argument("-e", "--end_date",
                        help="end date to harvest (YYYYMMDD)")
    parser.add_argument("-n", "--num_days", type=int,
                        help="number of days to harvest")
    args = parser.parse_args()
    return args

def read_dataset_conf(conf_fname, logger=None):
    """Read YAML config file.

    Args:
        conf_fname (str): Name of YAML configuration file.
        logger (logger): Logger object

    Returns:
        dict: Configuration parameters from the configuration file.
    """
    if os.path.isfile(conf_fname):
        with open(conf_fname, 'r') as f:
            try:
                conf = yaml.safe_load(f)
            except yaml.YAMLError as exc:
                print(exc)
    else:
        if logger:
            logger.error("{} does not exist.".format(conf_fname))
        sys.exit(5)
    return conf

def set_date_range(args, date_fmt="%Y%m%d", logger=None):
    """Determine desired start and end times to harvest from input parameters.

    Args:
        args (ArgumentParser): command line parameters
        date_fmt (str): Format of date part of times
        logger (logger): Logger object

    Returns:
        datetime: Start date to harvest.
        datetime: End date to harvest.
    """
    # Check validity of supplied arguments
    utcnow = datetime.utcnow()
    utc_today = datetime(utcnow.year, utcnow.month, utcnow.day,
                         tzinfo=timezone.utc)
    if args.get('start_date'):
        start_date_in = datetime.strptime(args.get('start_date'), date_fmt)
        start_date = datetime(start_date_in.year, start_date_in.month,
                              start_date_in.day, start_date_in.hour, 0, 0, tzinfo=timezone.utc)
        if start_date > utc_today:
            if logger is not None:
                logger.error("Cannot specify a start date in the future")
            sys.exit(1)
    if args.get('end_date'):
        end_date_in = datetime.strptime(args.get('end_date'), date_fmt)
        end_date = datetime(end_date_in.year, end_date_in.month,
                            end_date_in.day, end_date_in.hour, 59, 59, tzinfo=timezone.utc)
        if end_date < start_date:
            if logger is not None:
                logger.error("End date cannot be before start date.")
            sys.exit(2)
    if args.get('num_days') and args.get('num_days') < 1:
        if logger is not None:
            logger.error("Cannot specify less than 1 days to harvest")
        sys.exit(3)
        
    # Determine start and end dates to harvest from supplied arguments
    if args.get('num_days'):
        ndays = timedelta(days=args.get('num_days')) - timedelta(seconds=1)
        if args.get('start_date') and args.get('end_date'):
            # User improperly specified: -n N -s YYYYMMDD -e YYYYMMDD
            if logger is not None:
                logger.error("Cannot specify all 3 of start date, end date and number of days")
            sys.exit(4)
        elif args.get('start_date'):
            # User specified: -n N -s YYYYMMDD; calculate end_date
            end_date = start_date + ndays
        elif args.get('end_date'):
            # User specified: -n N -e YYYYMMDD; calculate start_date
            start_date = end_date - ndays
        else:
            # User specified: -n N
            # Only num_days specified; defaulting to that many days
            # ending on today.
            end_date = datetime(utc_today.year, utc_today.month, utc_today.day,
                                23, 59, 59, tzinfo=timezone.utc)
            start_date = end_date - ndays
    else:
        if args.get('start_date') and args.get('end_date'):
            # User specified -s YYYYMMDD -e YYYYMMDD; nothing more needs to
            # be done.
            pass
        elif args.get('start_date'):
            # User specified -s YYYYMMDD; set end date to today
            end_date = datetime(utc_today.year, utc_today.month, utc_today.day,
                                23, 59, 59, tzinfo=timezone.utc)
        elif args.get('end_date'):
            # User specified -e YYYYMMDD; set start date same as end date
            start_date = end_date
        else:
            # User specified no arguments; harvest just for today
            start_date = utc_today
            end_date = datetime(utc_today.year, utc_today.month, utc_today.day,
                                23, 59, 59, tzinfo=timezone.utc)
    return start_date, end_date

def replace_template(template, cur_date):
    """Replace format strings in template with appropriate date fields.

    Args:
        template (str): Template string with format identifiers
        cur_date (datetime): Date being harvested

    Returns:
        str: Template string with format substrings replaced by the
             appropriate fields in the specified date.
    """
    trans_key = {"%Y": "{:04d}".format(cur_date.year),
                 "%m": "{:02d}".format(cur_date.month),
                 "%d": "{:02d}".format(cur_date.day),
                 "%H": "{:02d}".format(cur_date.hour),
                 "%M": "{:02d}".format(cur_date.minute),
                 "%S": "{:02d}".format(cur_date.second)}
    formatted_str = template
    for key, val in trans_key.items():
        formatted_str = formatted_str.replace(key, val)
    return formatted_str

def time_setting_dict(time_str):
    """Convert a time string like 90s, 3h, 1d, to a dictionary with a
    single keyword-value pair appropriate for keyword value settings to
    the python datetime.timedelta function.  For example, input of "3h"
    should return {"hours": 3}.

    Args:
        time_str: Time string with units (e.g., 90s, 3h, 1d)

    Returns:
        dict: Keyword-value pair indicating command line arguments
    """
    time_unit_dict = {"s": "seconds",
                      "m": "minutes",
                      "h": "hours",
                      "d": "days",
                      "w": "weeks"}
    return {time_unit_dict[time_str[-1]]: int(time_str[:-1])}

def paths_generator(start_date, end_date, local_basedir, dataset_conf):
    """Generator that yields remote url, local directory and local path for
    the download.

    Args:
        start_date (datetime): Start date/time to harvest
        end_date (datetime): End date/time to harvest
        local_basedir (str): Directory to download to
        dataset_conf (dict): Dataset configuration dictionary

    Yields:
        str: Remote url
        str: Local file name
    """
    time_res = dataset_conf["time_res"]
    time_incr = timedelta(**time_setting_dict(time_res))
    cur_date = start_date
    while cur_date <= end_date:
        local_fname = replace_template(dataset_conf["local_path_template"],
                                       cur_date)
        url = replace_template(dataset_conf["url_template"], cur_date)
        local_path = os.path.join(local_basedir, local_fname)
        yield url, local_path, local_fname
        cur_date += time_incr

def harvest_date_range(start_date, end_date, local_basedir,
                       dataset_conf, aws=False, logger=None):
    """Retrieve granules in the specified time range.

    Args:
        start_date (datetime): Start date/time to harvest
        end_date (datetime): End date/time to harvest
        local_basedir (str): Directory to download to
        dataset_conf (dict): Dataset configuration dictionary
        logger (logger): Logger object
    """
    print('harvest_date_range')
    if aws:
        import boto3
        s3 = boto3.resource("s3")
        target_bucket = s3.Bucket(dataset_conf["target_bucket_name"])

    for url, local_path, local_fname in paths_generator(start_date, end_date,
                                           local_basedir, dataset_conf):
        local_dir = os.path.dirname(local_path)
        if not os.path.exists(local_dir):
            os.makedirs(local_dir)
        if not os.path.exists(local_path):
            try:
                print()
                print('url is {}'.format(url))
                print('local path is {}'.format(local_path))
                urllib.request.urlretrieve(url, local_path)
                if aws:
                    try:
                        target_bucket.upload_file(local_path, "{}/{}".format(dataset_conf["target_bucket_folder"],local_fname))
                        logger.warning("Uploaded granule to {}/{}".format(dataset_conf["target_bucket_folder"],local_fname))

                    except:
                        logger.error("Unable to upload {} to S3".format(local_fname))
            except:
                logger.error("Unable to download {}".format(url))
            else:
                logger.warning("Downloaded {} to {}".format(url, local_path))
  
def main(ds_name="", aws=False,event_time=None):
    print('on AWS {}'.format(str(aws)))
    """Main program.  Parse arguments, and harvest the requested dates from
    the remote archive.
    """
    # Initializer logger
    logger = logging.getLogger()
    args = {}

    if not aws:
        # Parse arguments and set date range to harvest
        args = vars(parse_args()) # convert namespace object to dict
        # ds_name = args.get('dataset_name')
        # print(args)

    else:
        args = {'start_date':event_time.get('start'), 'end_date':event_time.get('end'), 'num_days':None}
        # print(args)

    print(args)
    local_basedir = '/tmp/' if aws else args.get('data_basedir')

    # set start/end dates
    date_fmt_precise = "%Y-%m-%dT%H:%M:%SZ"
    date_fmt = "%Y%m%d" if not aws else date_fmt_precise
    start_date, end_date = set_date_range(args, date_fmt=date_fmt, logger=logger)
    start_date_str = start_date.strftime(date_fmt_precise)
    end_date_str = end_date.strftime(date_fmt_precise)
    print('start date {}'.format(start_date_str))
    print('end date {}'.format(end_date_str))
    logger.info("Harvesting between {} and {} to {}".format(start_date_str,
                                                                end_date_str,
                                                                local_basedir))
   
    # Read dataset configuration
    # dataset_rtma_15min_noaa.yaml
    conf_file_relpath = ".cedas/"
    conf_file_relpath += "{}.yaml".format(ds_name) if aws else "dataset.yaml"

    conf_fname =  conf_file_relpath if aws else os.path.join(local_basedir, conf_file_relpath)
    dataset_conf = read_dataset_conf(conf_fname, logger=logger)
    
    # Harvest data for the specified date range from the remote archive.
    harvest_date_range(start_date, end_date, local_basedir, dataset_conf, aws, 
                       logger=logger)

def lambda_handler(event, context):
    main(event['dataset'],aws=True,event_time={'start':event['start_date'],'end':event['end_date']})
    return {
        'statusCode': 200,
        'body': json.dumps('Done harvesting!')
    }

if __name__ == "__main__":
    main()
    # for simulating test on AWS locally
    # event = {
    #     "dataset": "rtma_15min_noaa",
    #     "start_date": "2020-05-29T00:00:00Z",
    #     "end_date": "2020-05-29T00:59:59Z"
    # }
    # lambda_handler(event,{})
