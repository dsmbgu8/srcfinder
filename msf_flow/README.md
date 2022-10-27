# msf_flow
End-to-end workflow for products related to the Methane Source Finder (MSF) portal.  This is organized into the following main modules:

### harvester

Automatically harvest (identify, download and ingest) data granules from a
remote archive.  This capability is general, but is applied to various
wind datasets in this project.

### workflow

Orchestrate top level workflow to generate data in a CSV table including
plume information, wind statistics, emission rate and uncertainty.

### wind_processor

Given a CSV table of data, compute wind statistics averaging over
time and space.

### plume_processor

Post-process and filter the plume list.

### utils

General utility functions of use in the other modules.


## Setup
Create a conda environment:
```
conda create -n m2af
```

Activate environment and install conda packages:
```
conda activate m2af
conda install -f conda-requirements.txt
```

Install pip packages:
```
pip install -r requirements.txt
```
