# msf_flow.harvester

Automatically harvest (identify, download and ingest) data granules from a
remote archive.  This capability is general, but is applied to various
wind datasets in this project.  As described below, parameters in a
configuration file are used by the automatic data harvester to resolve
paths to a dataset on a remote hosting site and to indicate local naming
conventions for the files.

## How to set up a dataset for harvesting

To harvest a dataset, do the following:

1. Create a directory to hold the data.
2. Inside that directory you just created, create a hidden
subdirectory called `.cedas`.
3. Inside that `.cedas` directory you just created, create a YAML configuration
file called `dataset.yaml`.  The values in `dataset.yaml` may be constructed
using the string substitution templates for elements of date or time,
as described below.  You must define the following required keywords in
`dataset.yaml`:
    * `url_template`: Web address of remote data granules
    * `local_path_template`: Desired local path for data granules
    * `time_res`: Time increment between data granules
4. Run `harvest.py` on the dataset directory (see example below)

## Template string substitution in the configuration

In the `url_template` and `local_path_template` settings the following
substrings may be used to indicate date- and time-specific string
substitutions as follows:

Template Substring | Granule Level Substitution
-------------------|----------------------------------------
`%Y` | 4-digit year in `{0000, 0001, ...}` (example: `2019`)
`%m` | 2-digit month in `{01, 02, ..., 12}`
`%d` | 2-digit day of month in `{01, 02, ...}`
`%H` | 2-digit hour in `{00, 01, ..., 23}`
`%M` | 2-digit minute in `{00, 01, ..., 59}`
`%S` | 2-digit second in `{00, 01, ..., 59}`

For the `time_res` setting, use an integer number of one of the
following units:

Template Substring | Units
-------------------|-------
`s` | seconds
`m` | minutes
`h` | hours
`d` | days
`w` | weeks

## Command line usage

```
usage: python3 harvest.py [-h] [-s START_DATE] [-e END_DATE] [-n NUM_DAYS] data_basedir

positional arguments:
  data_basedir          path to local dataset directory

optional arguments:
  -h, --help            show this help message and exit
  -s START_DATE, --start_date START_DATE
                        start date to harvest (YYYYMMDD)
  -e END_DATE, --end_date END_DATE
                        end date to harvest (YYYYMMDD)
  -n NUM_DAYS, --num_days NUM_DAYS
                        number of days to harvest
```

Any sensible combination of the `-s`, `-e`, or `-n` parameters can be used
to specify the desirered date range to harvest:

### Example: Specify start and end dates:

```
python3 harvest.py -s 20190921 -e 20190927 /data/mydata
```

### Example: Specify start date and number of days:

```
python3 harvest.py -s 20190921 -n 7 /data/mydata
```

### Example: Specify end date and number of days:

```
python3 harvest.py -e 20190927 -n 7 /data/mydata
```

### Example: Specify number of days ending today:

```
python3 harvest.py -n 7 /data/mydata
```

### Example: Specify just a single day (today):

```
python3 harvest.py /data/mydata
```

