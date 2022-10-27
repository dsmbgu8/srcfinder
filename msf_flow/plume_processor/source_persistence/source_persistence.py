####################################################################
#
#                 calculating source persistence
# -----------------------------------------------------------------
#
#     Author:  Kelsey Foster
#     Translated from R to Python by Elizabeth Yam
#     This script was written to calculate source persistence.
#
#####################################################################

import json
import sys
import csv
import pandas as pd
import geopandas as gpd
import os
import boto3
import fiona

# USER DEFINED VARIABLES---------------------------------------------
# flightlines = '/Users/eyam/M2AF/MSF_manual_data_workflow/scripts/2_source_persistence_calculation/source_persistence/flightlines_all_20181105.shp' # fall 2016 and 2017 flightlines footprints
# source_list = '/Users/eyam/M2AF/MSF_manual_data_workflow/scripts/2_source_persistence_calculation/source_persistence/plume-list-20190413.csv' # source list
# outpath = '/Users/eyam/M2AF/MSF_manual_data_workflow/scripts/2_source_persistence_calculation/source_persistence/' 
# # outfile = 'source_persistence_output_v20190415.csv'
# outfile = 'source_persistence_output_v20190415_v2.csv'


# source.persistence FUNCTION----------------------------------------
# Description: calculates the persistence of CH4 detections at CH4 sources by
#                 1. counting the number of plumes detected at each source
#                 2. finding the intersection between sources and flightlines 
#                 3. counting the number of flightlines that intersect each source
#                 4. calculating persistence of CH4 detection for each source: total 
#                    CH4 observations from plume list/total flightlines

# Inputs:
#     source.list - plume list  
#     flightlines - shapefile of flight lines  
#     outpath - folder where the csv is to be exported
#     outfile - name of output file with .csv extension

# Output: CSV with source persistence 

# Assumptions: input shapefile has a "Source.identifer" column, assume shapefile does not have self-intersecting polygons
# note: if function fails, it is likely due to changes in columns names or order of columns

def read_plume_list(source_list):
    sources = []
    headers = []
    with open (source_list) as f:
        reader = csv.reader(f) 
        skip = True
        for row in reader:
            # skip 1st row, which is column headers
            if skip:
                skip = False
                headers = row
            else:
                sources.append(row)
    print('read plumelist csv')
    return sources,headers

def plumes_per_source(sources, id_index=0):
    # get unique source points with lat/lon
    # Col A / index 0: Source ID
    # Col C / index 2: Plume Longitude (deg)
    # Col B / index 1: Plume Latitude (deg)
    unique_srcs = []
    # source_pts = {}
    plume_freq = {}
    source_to_plumes = {}
    for source in sources:
        source_id = source[id_index]
        if source_id in plume_freq.keys():
            source_to_plumes[source_id].append(source)
            plume_freq[source_id] = plume_freq[source_id] + 1
        else:
            unique_srcs.append(source)
            source_to_plumes[source_id] = [source]
            plume_freq[source_id] = 1
    return plume_freq,unique_srcs,source_to_plumes

def flightline_per_source(flightline_names,flightline_geo,source_ids,source_geos):
    flightlines_per_source = {}
    flightline_freq = {}
    for i in range(len(source_geos)):
        src_id = source_ids[i]
        src = source_geos[i]
        # pt = src.geometry
        for j in range(len(flightline_geo)):
            flight_name = flightline_names[j]
            geo = flightline_geo[j]
            # if flight contains source
            if src.within(geo):
                if src_id in flightlines_per_source.keys():
                    flightlines_per_source[src_id].append(flight_name)
                    flightline_freq[src_id] = flightline_freq[src_id] + 1
                else:
                    flightlines_per_source[src_id] = [flight_name]
                    flightline_freq[src_id] = 1
    
    return flightline_freq,flightlines_per_source

def source_persistence(source_list, flightlines, output_path, aws=False, target_bucket=None):
    # if file from s3, download source list to tmp and set f to that path
    if aws:
        # download source_list file
        source_list_localp = '/tmp/' + source_list.split('/')[-1]
        print('download source list file {} to {}'.format(source_list,source_list_localp))
        target_bucket.download_file(source_list,source_list_localp)
        source_list = source_list_localp

        # download flightlines file
        flightlines_localp = '/tmp/' + flightlines.split('/')[-1]
        print('download flightlines file {} to {}'.format(flightlines,flightlines_localp))
        target_bucket.download_file(flightlines,flightlines_localp)
        
        # need dbf file for attributes
        flightlines_dbf = flightlines[:-3]+'dbf'
        flightlines_dbf_localp = flightlines_localp[:-3]+'dbf'
        target_bucket.download_file(flightlines_dbf,flightlines_dbf_localp)
  
        # need shx file for attributes
        flightlines_dbf = flightlines[:-3]+'shx'
        flightlines_dbf_localp = flightlines_localp[:-3]+'shx'
        target_bucket.download_file(flightlines_dbf,flightlines_dbf_localp)
        
        flightlines = flightlines_localp

    # read flightlines shapefile into geodataframe
    flightlines_gdf = gpd.read_file(flightlines)
    print('read flightlines shapefile into geodataframe')
    # make sure gdf file read attributes
    print(flightlines_gdf.head())

    # if flightlines_gdf.Flight_Run does not exists, calculate it from Name column:
    if 'Flight_Run' not in flightlines_gdf:
        flightlines_gdf['Flight_Run']=[flightlines_gdf['Name'][i].split()[0] for i in range(len(flightlines_gdf['Name']))]

    # read in plume list
    sources,headers = read_plume_list(source_list)
    headers = [header.strip() for header in headers]

    id_index=0
    if 'Source identifier' in headers:
        id_index= headers.index('Source identifier')
    elif 'Source ID' in headers:
        id_index= headers.index('Source ID')

    ### 1. count number of plumes at each source
    plume_freq, unique_src_pts, source_to_plumes = plumes_per_source(sources, id_index)

    # create data frame of unique source points
    dct = {}
    for i in range(len(headers)):
        dct[headers[i]] = [source[i] for source in unique_src_pts]

    source_df = pd.DataFrame(dct)
    print('created plume source dataframe')

    # convert source dataframe to geodataframe
    source_gdf = gpd.GeoDataFrame(
        source_df, 
        geometry=gpd.points_from_xy(source_df['Plume Longitude (deg)'], source_df['Plume Latitude (deg)'])
    )
    print('converted source dataframe to geodataframe')

    ### 2. find the intersection between sources and flightlines
    # (unnecessary step skipped)

    ### 3. count the number of flightlines that intersect each source
    # count overflights per unique source and save flightnames for each source
    flightline_freq, flightlines_per_source = flightline_per_source(flightlines_gdf.Flight_Run,flightlines_gdf.geometry,source_gdf['Source identifier'],source_gdf.geometry)
    print('counted flightlines per source')

    ### 4. calculate persistence of CH4 detection: total CH4 observations from plume list/total flightlines
    source_persistence = {}
    unique_src_ids = source_df['Source identifier']
    missing_src_ids = []

    for i in range(len(unique_src_ids)):
        src_id = unique_src_ids[i]
        #flightline_freq missing the src_id, set the value to NAN
        if src_id not in flightline_freq:
            source_persistence[src_id]=float("NAN")
            missing_src_ids.append(src_id)
            continue
        source_persistence[src_id] = plume_freq[src_id] / flightline_freq[src_id]
    print('calculated source persistence')

    source_df = pd.DataFrame(source_gdf.drop(columns='geometry'))
    # add calculations as extra columns
    source_df['observed.plumes'] = [plume_freq[src_id] for src_id in unique_src_ids]
    #source_df['total.overflights'] = [flightline_freq[src_id] for src_id in unique_src_ids]
    source_df['total.overflights'] = [flightline_freq[src_id] if src_id not in missing_src_ids else float("NAN") for src_id in unique_src_ids] 
    source_df['source.Persistence'] = [source_persistence[src_id] for src_id in unique_src_ids]
    print('appended calculated columns')

    # write to file
    if aws:
        output_localp = '/tmp/'+output_path.split('/')[-1]
        print('writing output file to {}'.format(output_localp))
        with open(output_localp, 'w') as f:
            source_df.to_csv(path_or_buf=f,index=True)
            print('updated dataframe written to csv')
        
        target_bucket.upload_file(output_localp,output_path)
        print('output csv {} uploaded to s3 {}'.format(output_localp,output_path))
    else:
        with open(output_path, 'w') as f:
            source_df.to_csv(path_or_buf=f,index=True)
        print('updated dataframe written to csv {}'.format(output_path))

# should have flightlines in predefined s3 bucket
# output to predefined s3 bucket

def lambda_handler(event, context):
    # triggered by plume list (source_list)
    if 'Records' not in event:
        print(event)

    bucket = event['Records'][0]["s3"]["bucket"]["name"]
    filename = event['Records'][0]["s3"]["object"]["key"]
    source_list = filename

    flightlines = os.environ["SHAPE_FILE_LOCATION"]  # to be updated
    output_path = 'data/source_persistence/source_persistence_{}'.format(os.path.basename(filename))

    s3 = boto3.resource("s3")
    target_bucket = s3.Bucket(bucket)

    source_persistence(source_list,flightlines,output_path,True,target_bucket)
    return {
        'statusCode': 200,
        'body': json.dumps('Source Persistence Lambda run successfully!')
    }
    

if __name__ == "__main__":
    flightlines = sys.argv[1]
    source_list = sys.argv[2]
    output_path = sys.argv[3]
    source_persistence(source_list,flightlines,output_path)
