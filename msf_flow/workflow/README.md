# msf_flow.workflow

Orchestrate top level workflow to generate data in a CSV table including
plume information, wind statistics, emission rate and uncertainty.
This module can be run from the command line or in other python code using
the API described below.  Only python3 is supported.

## Command-Line Usage

```
usage: python3 msf_flow.py [-h] -p PLUMEDIR [-r REGEX] -w WINDIR -o OUTFILE [-f]

optional arguments:
  -h, --help            show this help message and exit
  -p PLUMEDIR, --plumedir PLUMEDIR
                        path to input plume file directory
  -r REGEX, --regex REGEX
                        Regular expression to match for plume files
  -w WINDIR, --windir WINDIR
                        path to input wind file directory
  -o OUTFILE, --outfile OUTFILE
                        path to output plume list
  -f, --force           Force reprocessing of all files (not just the new ones)
```

### Example: Detect and process only new plume files
Traverse the plume directory tree and process just the plume files that are
new since the last time this plume directory was processed.

```
python3 /src/msf_flow/workflow/msf_flow.py -p /data/plumes -r 'plumes.*.csv$' -w /data/winds -o /data/results/plumes_out.csv
```

### Example: Detect and process all plume files
Traverse the plume directory tree and use the `-f`/`--force` flag to process
all plume files found regardless of whether or not they were previously
processed.

```
python3 /src/msf_flow/workflow/msf_flow.py -p /data/plumes -r 'plumes.*.csv$' -w /data/winds -op /data/results/plumes_out.csv -f
```

## Python3 API:

- **process_plumes**(*file_list, winds_dir, fill=None, logger=None*)

argument    | type                | required? | description
------------|---------------------|-----------|--------------------------------
file_list   | list of str         | required  | list of plume files 
winds_dir   | str                 | required  | path to wind datasets
fill        | str \| int \| float | optional  | fill value to use as default
logger      | logging.Logger      | optional  | message logger

- **insert_plumes_in_file**(*plumes, out_fname, sort_by_key=sort_by_key, logger=logger*)

argument    | type                | required? | description
------------|---------------------|-----------|--------------------------------
plumes      | OrderedDict         | required  | plume data
out_fname   | str                 | required  | output plume list file name
sort_by_key | str                 | optional  | key by which to sort plume list
logger      | logging.Logger      | optional  | message logger

### Generic Example
```
# Import module
from msf_flow.workflow import msf_flow

# Configuration
...
plume_files = ...
wind_dir = ...
out_fname = ...
sort_by_key = ...
...

# Process plumes
plumes = msf_flow.process_plumes(plume_files, winds_dir,
                                 fill=-9999, logger=logger)

# Insert plumes into output file
insert_plumes_in_file(plumes, out_fname,
                      sort_by_key=sort_by_key, logger=logger)
```

### Example: Unsorted plume list

```
from msf_flow.workflow import msf_flow

plume_files = ["/data/plumes/plumes1.csv",
               "/data/plumes/plumes2.csv",
               "/data/plumes/plumes3.csv"]
wind_dir = "/data/winds"

plumes = msf_flow.process_plumes(plume_files, wind_dir)
```

### Example: Sorted plume list

```
...
sort_by_key = "# Candidate ID"

plumes = msf_flow.process_plumes(plume_files, wind_dir, sort_by_key=sort_by_key)
```
