import argparse
from csv import DictReader, DictWriter
import re
import sys
from datetime import datetime, timedelta
import numpy as np
from collections import OrderedDict
from wind_processor.windspeed import stationWindSpeed

candidate_id_key = "Candidate ID"

def parse_args():
    """Retrieve command line parameters.
    
    Returns:
        ArgumentParse: command line parameters
    """
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("plumes", help="path to input plume file")
    parser.add_argument("output", help="path to output plume file")
    parser.add_argument("token", help="token for access to station data")
    args = parser.parse_args()
    return args.plumes, args.output, args.token

def read_plumes(plume_fname):
    with open(plume_fname, 'r') as fin:
        reader = DictReader(fin)
        plume_list = list(reader)
    return plume_list


def get_datetime_from_str(s, dt_regex="(\d{8})t(\d{4})",
                          dt_fmt="%Y%m%d%H%M"):
    pattern = re.compile(dt_regex)
    m = pattern.search(s)
    if m is None:
        raise ValueError("No match for {} found in {}".
                         format(dt_regex, s))
    if len(m.groups()) != 2:
        raise ValueError("Could not get date and time from {} with regex {}".
                         format(s, dt_regex))
    dt_str = ''.join(m.groups())
    dt = datetime.strptime(dt_str, dt_fmt)
    return dt


def get_station_data_for_plume(plume,
                               cand_id_key=candidate_id_key,
                               lat_key="Plume Latitude (deg)",
                               lon_key="Plume Longitude (deg)",
                               fill="-9999",
                               dt_regex = "(\d{8})t(\d{4})",
                               dt_fmt="%Y%m%d%H%M",
                               delta_mins=5, radius_km=20,
                               variables="wind_speed", token=None):
    cand_id = plume[cand_id_key]
    dt = get_datetime_from_str(cand_id, dt_regex=dt_regex)
    start_dt = dt - timedelta(minutes=delta_mins)
    end_dt = dt + timedelta(minutes=delta_mins)
    start_dt_str = datetime.strftime(start_dt, dt_fmt)
    end_dt_str = datetime.strftime(end_dt, dt_fmt)
    station_data = stationWindSpeed(plume[lon_key], plume[lat_key],
                                    start_dt_str, end_dt_str, radius_km,
                                    variables=variables, token=token)
    dist = station_data[1]
    if np.isnan(dist):
        dist = fill
    windspeed = station_data[0]
    if np.isnan(windspeed):
        windspeed = fill
    d = OrderedDict()
    d["Distance to Nearest Station (km)"] = dist
    d["Average Windspeed at Nearest Station (m/s)"] = windspeed
    d["Station search radius (km)"] = radius_km
    d["Station search time delta (+/- minutes)"] = delta_mins
    return d

def get_station_data_for_plumes(plume_list,
                                cand_id_key=candidate_id_key,
                                lat_key="Plume Latitude (deg)",
                                lon_key="Plume Longitude (deg)",
                                fill="-9999", token=None):

    plume_list_with_station_data = \
        [plume.update(get_station_data_for_plume(plume, token=token)) or plume
         for plume in plume_list]
    return plume_list_with_station_data


def main():
    # Parse command line arguments.
    plume_fname, out_fname, token = parse_args()

    # Read plume list.
    plume_list = read_plumes(plume_fname)

    # Get station data.
    plume_list_with_station_data = get_station_data_for_plumes(plume_list,
                                                               token=token)

    # Get field names from the first plume
    field_names = list(plume_list_with_station_data[0].keys())

    # Write processed plumes to output file
    with open(out_fname, 'w') as fout:
        writer = DictWriter(fout, fieldnames=field_names)
        writer.writeheader()
        for plume in plume_list_with_station_data:
            writer.writerow(plume)
    print("Plume file with source identification written to {}".
          format(out_fname))
    

if __name__ == "__main__":
    main()
