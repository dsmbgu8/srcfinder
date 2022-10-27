import os
import glob
import re
import argparse
import itertools
from collections import OrderedDict
from multiprocessing import Pool
from shutil import copyfile
from functools import partial
from csv import DictReader, DictWriter
import logging
import datetime
from dateutil.tz import tzlocal

from wind_processor.running_windspeed \
    import compute_wind_stats, compute_emission_rate
from wind_processor.station_winds import get_station_data_for_plume
from wind_processor.wind_type import WindType
from utils.dir_watcher import DirWatcher
from utils.logger import init_logger

TOKEN = os.getenv('MESONET_API_TOKEN', default=None)
AWS = 'AWS' in os.environ.keys() and os.environ['AWS'] == 'TRUE' else False
if AWS:
    import boto3

def parse_args(default_minppmm=1000):
    """Retrieve command line parameters.

    Returns:
        ArgumentParse: command line parameters
    """
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("-p", "--plumedir", required=True,
                        help="path to input plume file directory")
    parser.add_argument("-r", "--regex", required=False,
                        default="ang.*_detections/ime_minppmm{}/ang.*_ime_minppmm{}.*".format(
                            default_minppmm, default_minppmm),
                        help="Regular expression to match for plume files")
    parser.add_argument("-w", "--windir", required=True,
                        help="path to input wind file directory")
    parser.add_argument("-o", "--outfile", required=True,
                        help="path to output plume list")
    parser.add_argument("-f", "--force",
                        help="Force reprocessing of all files (not just the new ones)",
                        action='store_true')
    parser.add_argument("-n", "--nprocs", type=int, default=1,
                        help="number of parallel processes to use; default=1 (sequential)")
    parser.add_argument("--flmode",
                        help="Executes script in flightline mode, running on a single flightline",
                        action="store_true")
    args = parser.parse_args()
    return (args.plumedir, args.regex, args.windir, args.outfile,
            args.force, args.nprocs, args.flmode)

def process_plume(winds_dir, plume, fill=None, logger=None):
    ''' Helper function to process each individual plume inside a plume list
    '''
    if logger is not None:
        logger.warning("processing plume {}".format(plume))
        logger.info(plume)

    # Compute wind and emission stats.
    emission_stats = OrderedDict()
    subdirs = []

    # get subdirs on s3
    if AWS:
        if 's3://' in winds_dir:
            bucket_key = winds_dir.split('s3://')[1]
            bucket = bucket_key.split('/')[0]
            winds_dir_cloud = '/'.join(bucket_key.split('/')[1:])

            s3_client = boto3.client('s3')
            resp = s3_client.list_objects(
                Bucket='bucket',
                Prefix='data/wind/',
                Delimiter='/'
            )
            lst = [e['Prefix'] for e in resp['CommonPrefixes']]
            subdirs = [e.split('/')[-2] for e in lst]
        else:
            if logger is not None:
                logger.warning('Please pass s3 path wind path for running on AWS (WINDIR={})'.format(winds_dir))
            else:
                print('Please pass s3 path wind path for running on AWS (WINDIR={})'.format(winds_dir))

    # get subdirs on local system
    else:
        subdirs = sorted([f for f in os.listdir(winds_dir)
                          if not f.startswith('.')])

    for subdir in subdirs:
        if logger is not None:
            logger.info("processing for winds in {}".format(subdir))
        wt = WindType(subdir)
        if not wt.is_unknown():
            wind_type = wt.type_as_str()
            winds_subdir = os.path.join(winds_dir, subdir)
            alts = sorted(wt.alts())

            # Add stats for gridded wind data at each relevant altitude
            # to plume.
            for alt in alts:
                wind_stats = compute_wind_stats(plume, winds_subdir,
                                                fill=fill, wind_type=wind_type,
                                                wind_alt=alt, logger=logger)
                plume.update(wind_stats)
            emission_stats.update(compute_emission_rate(plume, wind_type,
                                                        fill=fill,
                                                        logger=logger))

    # Add wind stastics from meteoroloogical stations to plume data.
    station_data = get_station_data_for_plume(plume, fill=fill, token=TOKEN)

    # Add emission statistics to plume data.
    plume.update(emission_stats)

    plume.update(station_data)
    if logger is not None:
        logger.info("Computed plume={}".format(plume))
    return plume

def get_minppmm_from_fname(fname):
    ''' Helper to grab minppmm value from the plume list filename
    '''
    regex_str = "minppmm(\d+)"
    pattern = re.compile(regex_str)
    match = pattern.search(fname)
    if match is None:
        raise ValueError("No match for {} found in {}".format(regex_str, fname))
    return int(match[1])

def dict_reader_plus_update(fname, d):
    ''' Helper to read in dicts from csv file.
    Automatically adds minppmm value for each plume
    '''
    with open(fname, 'r') as f:
        reader = DictReader(f, skipinitialspace=True)
        # The dictionary "update" method returns None, but I need the updated
        # dictionary in each row.  I am using the "or" operator below to
        # get that.  This works because "None or X" returns X.
        out = [row.update(d) or row for row in reader]
    return out

def process_plumes(flist, winds_dir, nprocs=1, fill=None,
                   minppmm_key="Minimum Threshold (ppmm)", logger=None):
    ''' Read all of the plumes from all of the files, add minimum PPMM threshold
    field to each plume (threshold value extracted from file name), and sort
    the plumes by "Candidate ID".
    '''
    if logger is not None:
        logger.info("flist={}".format(flist))
    plumes = list(itertools.chain.from_iterable(
        [dict_reader_plus_update(fname,
                                 {minppmm_key: get_minppmm_from_fname(fname)})
         for fname in flist]))
    if logger is not None:
        logger.info("plumes={}".format(list(plumes)))

    # Process plumes in parallel using partial function with all arguments
    # frozen except for input plume.
    process_plumes_part = partial(process_plume,
                                  winds_dir, fill=fill, logger=logger)
    pool = Pool(nprocs)
    plumes_ext = pool.map(process_plumes_part, plumes)

    # Get plume keys (field header names) from the first plume. Sort plume
    # according to the values for the first key (expected to be Candidate ID).
    sort_by_key = list(plumes_ext[0].keys())[0]
    plumes_ext = sorted(plumes_ext, key=lambda d: d[sort_by_key])
    if logger is not None:
        logger.info("Computed plumes={}".format(plumes_ext))
    return plumes_ext

def insert_plumes_in_file(plumes, fname, sort_by_key=None, logger=None):
    ''' Write processed plumes to output file.
    If output file already exists, insert plumes in sorted order
    '''
    if len(plumes) > 0:
        fname_local = fname
        # for aws usage
        bucket_name = ''
        key = ''
        if AWS:
            if 's3://' in fname:
                # check if output file already exists in s3
                print('checking if {} exists on s3'.format(fname))
                bucket_key = fname.split('s3://')[1]            # s3://bucket-name/path/to/file -> bucket-name/path/to/file
                bucket_name = bucket_key.split('/')[0]          # bucket-name/path/to/file -> bucket-name
                key = '/'.join(bucket_key.split('/')[1:])       # bucket-name/path/to/file -> path/to/file
                prefix = '/'.join(key.split('/')[:-1])          # path/to/file -> path/to
                s3_client = boto3.client('s3')
                resp = s3_client.list_objects(
                    Bucket=bucket_name,
                    Prefix=prefix
                )
                obj_lst = [e['Key'] for e in resp['Contents']]
                fname_local = '/tmp/'+key
                # if output file exists in s3, download it
                if fname in obj_lst:
                    print('downloading s3://{}/{} to {}'.format(bucket_name, fname, fname_local))
                    boto3.resource('s3').Bucket(bucket_name).download_file(fname, fname_local)
            else:
                if logger is not None:
                    logger.warning('Please pass s3 output path for running on AWS ({})'.format(fname))
                else:
                    print('Please pass s3 output path for running on AWS ({})'.format(fname))

        # If output file already exists, insert new plumes into it
        # in sorted order (sorted by first column).
        if os.path.isfile(fname_local):
            with open(fname_local, 'r') as fin:
                reader = DictReader(fin)
                plumes = list(reader) + plumes

            # Make backup copy of original plume file
            fname_bak = fname_local + '.bak'
            copyfile(fname_local, fname_bak)
            if logger is not None:
                logger.critical(
                    "Original plume file backed up to {}".format(fname_bak))

        # Sort plume list if sort_by_key is provided
        if sort_by_key is not None:
            if sort_by_key in plumes[0]:
                plumes = sorted(plumes, key=lambda d: d[sort_by_key])
            else:
                if logger is not None:
                    logger.warning("Sort key {} not found.".
                                   format(sort_by_key))
                    logger.warning("Plumes left unsorted.")

        # Get field names from the first plume
        field_names = list(plumes[0].keys())

        # if output path doesn't exist, create it
        fname_local_path = '/'.join(fname_local.split('/')[:-1])
        if not os.path.isdir(fname_local_path):
            os.makedirs(fname_local_path)

        # Write processed plumes to output file
        with open(fname_local, 'w') as fout:
            writer = DictWriter(fout, fieldnames=field_names)
            writer.writeheader()
            for plume in plumes:
                try:
                    writer.writerow(plume)
                except:
                    if logger is not None:
                        logger.warning(
                            "Could not write plume: {}".format(plume))
                        logger.warning(
                            "Check that the plume fields match the header in {}".format(fname))
        if logger is not None:
            logger.critical("Extended plume file written to {}".format(fname))

        # upload file to S3 if using AWS
        if AWS:
            print('uploading {} to {}'.format(fname_local,fname))
            boto3.resource('s3').Bucket(bucket_name).upload_file(fname_local,key)
        if logger is not None:
            logger.warning("Extended plume file uploaded to {}".format(fname))

    else:
        if logger is not None:
            logger.warning("Skipped insertion because plume list was empty")

def main():
    ''' Main function. Parses arguments, finds input plume file(s) and
    runs process plumes helper functions to find or calculate extra metadata and stats.
    '''
    # Initializer logger
    logger = init_logger()

    # ---------------------
    # set process arguments
    # ---------------------
    # Parse command line arguments.
    (plume_dir, plume_file_regex, winds_dir, out_fname,
     force, nprocs, flmode) = parse_args()

    # ---------------------
    # find plume files
    # ---------------------
    if AWS:
        plume_local_dir = '/tmp/data/plumes'
        winds_local_dir = '/tmp/data/winds'
        plume_bucket = 'bucket'
        fname = plume_dir.split('/')[-1]
        if 's3://' in plume_dir:
            # s3://bucket-name/plume-dir/file -> bucket-name/plume-dir/file
            bucket_key = plume_dir.split('s3://')[1]
            # bucket-name/plume-dir/file -> bucket-name/plume-dir
            bucket_dir = '/'.join(bucket_key.split('/')[:-1])
            # bucket-name/plume-dir -> bucket-name
            plume_bucket = bucket_dir.split('/')[0]
            # bucket-name/plume-dir -> plume-dir
            plume_local_dir = '/tmp/'+'/'.join(bucket_dir.split('/')[1:])
            if not os.path.isdir(plume_local_dir):
                os.makedirs(plume_local_dir)
        else:
            warn_m = 'Please pass s3 plume path for running on AWS ({})'.format(plume_dir)
            if logger is not None:
                logger.warning(warn_m)
            else:
                print(warn_m)

        if 's3://' in winds_dir:
            # s3://bucket-name/winds-dir -> winds-dir
            winds_local_dir = '/tmp/'+'/'.join(winds_dir.split('s3://')[1].split('/')[1:])
            if not os.path.isdir(winds_local_dir):
                os.makedirs(winds_local_dir)
        else:
            warn_m = 'Please pass s3 winds path for running on AWS ({})'.format(winds_dir)
            if logger is not None:
                logger.warning(warn_m)
            else:
                print(warn_m)

        # set output filename
        out_fname = out_fname+'/'+fname

        # pass single new plume file as plumedir arg when on AWS
        plume_local = plume_local_dir+'/'+ fname
        plume_key = plume_local.split('/tmp/')[1]

        # download the plume file
        # bucket_name = plume_dir
        s3_resource = boto3.resource('s3')
        bucket = s3_resource.Bucket(plume_bucket)
        print('downloading s3://{}/{} to {}'.format(plume_bucket, plume_key, plume_local))
        bucket.download_file(plume_key, plume_local)
        new_files = [plume_local]

    elif flmode:
        # In flmode the plume_dir is the directory containing the plume file
        # and the plume_file_regex is the filename, so this results in a single match
        new_files = glob.glob(os.path.join(plume_dir, plume_file_regex))
    else:
        args = {
            "force": force,
            "regex": plume_file_regex
        }

        # Check for new files matching indicated regex.
        watcher = DirWatcher(plume_dir, **args)
        new_files = watcher.whats_new_local()

    if logger is not None:
        logger.warning("Found {}".format(new_files))

    # ---------------------
    # process plume files
    # ---------------------
    # If new files are found, process each plume in the files.
    if len(new_files) > 0:
        # Process each plume.  We compute extra values for each plume
        # including wind statistics and emission rate and uncertainty.
        plumes = process_plumes(new_files, winds_dir, nprocs=nprocs,
                                fill=-9999, logger=logger)

        # Insert new plumes into output file.
        insert_plumes_in_file(plumes, out_fname,
                              sort_by_key="Candidate ID",
                              logger=logger)
    else:
        logger.warning("Nothing to do")

if __name__ == "__main__":
    if AWS:
        main()
        print('on AWS')

    else:
        main()
