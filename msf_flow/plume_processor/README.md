# msf_flow.plume_processor

Post-process and filter the plume list.

## Source Persistence
`source_persistence.py` is `source_persistence.R` translated into Python3 and parameterized for to take commandline arguments.  It calculates:
1. number of plumes detected at each source,
2. number of overflights at each source, and 
3. the source persistence of each source (# plumes / # overflights)

### Usage
``` bash
python source_persistence.py <flightlines_shapefile.shp> <plume_list.csv> <output_file>
```

Example:
``` bash
python source_persistence.py flightlines_all_20181105.shp plume-list-20190413.csv source_persistence_output_v20190415.csv
```

### Required Libraries
- csv
- pandas
- geopandas
