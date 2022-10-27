###################################################################
#
#      removing duplicate/overlapping plumes from plume list
# -----------------------------------------------------------------
#
#     Author:  Kelsey Foster
#     Translated from R to Python by Elizabeth Yam
#     This script was written to eliminate the possibility of
#     double counting emissions from methane plumes within in
#     the same search radius (maximum fetch = 150m). Please see
#     CA's Methane Super-emitters (Duren etal) supplementary
#     material (SI S2.5 and S2.8) for additional information.
#
####################################################################

# USER DEFINED VARIABLES---------------------------------------------
#df_path = '/Users/eyam/M2AF/MSF_manual_data_workflow/scripts/1_plume_proximity_filtering/flux_ovest_input_data.csv' #plume list as csv
#out_path = "/Users/eyam/M2AF/MSF_manual_data_workflow/scripts/1_plume_proximity_filtering/flux_overest_output_08102020.csv" #output file

import os
import json
import sys
import csv
import shapely as sp
import numpy as np
import pandas as pd
import geopandas as gp
import boto3
import traceback
from datetime import datetime




# defaults
max_overlap_default:float = .30 #max allowable overlap between plume search radii (should be decimal)
geog_crs_default:str ='+proj=longlat +datum=WGS84 +no_defs +ellps=WGS84 +towgs84=0,0,0'
proj_crs_default:str ='+proj=aea +lat_1=34 +lat_2=40.5 +lat_0=0 +lon_0=-120 +x_0=0 +y_0=-4000000 +ellps=GRS80 +datum=NAD83 +units=m +no_defs'

def parse_args():
    """Retrieve command line parameters.
    
    Returns:
        ArgumentParse: command line parameters
    """
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("plumes", help="path to input plume file")
    parser.add_argument("output", help="path to output plume file")
    parser.add_argument("--max_overlap", default = .30,
                        help="max_overlap value, default 0.3")
    
    args = parser.parse_args()
    return args.plumes, args.output, float(args.max_overlap)

def read_plume_list(source_list:str):
    ''' Reads in initial plume list CSV file and converts to a 2D string array for processing.
    Inputs:
        source_list (string): Path to initial plume list CSV file.

    Returns:
        sources (string[][]): Data in list form, to be converted into Pandas DataFrame.
        headers (string[]): :ist of column headers.
    '''
    sources = []
    headers = []
    try:
        # assume utf-8 encoding
        with open (source_list,'r') as f:
            print("source_list : {}".format(source_list))
            reader = csv.reader(f) 
            skip = True
            for row in reader:
                # skip 1st row, which is column headers
                if skip:
                    skip = False
                    headers = row
                else:
                    sources.append(row)
        print("Final Length of sources UTF-8: {}".format(len(sources)))
        return sources,headers
    except Exception as e1:
        print("Error with utf-8 encoding : {}".format(str(e1)))
        traceback.print_exc()
        try:
            # try ISO-8859-1 encoding (Windows)
            sources = []
            headers = []
            with open (source_list,'r',encoding = "ISO-8859-1") as f:
                print("source_list ISO-8859-1: {}".format(source_list))
                reader = csv.reader(f) 
                skip = True
                print('WARNING: CSV was decoded using \'ISO-8859-1\'')
                for row in reader:
                    # skip 1st row, which is column headers
                    if skip:
                        skip = False
                        headers = row
                    else:
                        sources.append(row)
            print("Final Length of sources ISO-8859-1 : {}".format(len(sources)))
            #print("headers : \n{}".format(headers))
            #print("sources : \n{}".format(sources))
 
            return sources,headers
        except Exception as e2:
            print("Error with ISO-8859-1 : {}".format(str(e2)))
            traceback.print_exc()
            try:
                # try mac_roman encoding
                sources = []
                headers = []
                with open (source_list,'r',encoding='mac_roman') as f:
                    print("source_list mac_roman: {}".format(source_list))
                    reader = csv.reader(f) 
                    skip = True
                    print('WARNING: CSV was decoded using \'mac_roman\'')
                    for row in reader:
                        # skip 1st row, which is column headers
                        if skip:
                            skip = False
                            headers = row
                        else:
                            sources.append(row)
                print("Final Length of sources mac_roman: {}".format(len(sources)))
                #print("headers : \n{}".format(headers))
                #print("sources : \n{}".format(sources))
                return sources,headers
            except Exception as e3:

                print("Error with mac_roman encoding : {}".format(str(e3)))
                traceback.print_exc()
                print('error decoding CSV file')
                return read_plume_list2(source_list)

def read_plume_list2(source_list:str):
    ''' Reads in initial plume list CSV file and converts to a 2D string array for processing.
    Inputs:
        source_list (string): Path to initial plume list CSV file.

    Returns:
        sources (string[][]): Data in list form, to be converted into Pandas DataFrame.
        headers (string[]): :ist of column headers.
    '''
    sources = []
    headers = []
    result = {}

    with open(source_list, 'rb') as rawdata:
        result = chardet.detect(rawdata.read(10000))

    if 'encoding' in result:
        with open (source_list,'r',encoding = result['encoding']) as f:
            reader = csv.reader(f)
            data = list(reader)
            skip = True
            for i in range(len(data)):
                try:
                    row = data[i]
                    # skip 1st row, which is column headers
                    if skip:
                        skip = False
                        headers = row
                    else:
                        sources.append(row)
                        print(row)
                except Exception as e:
                    print(str(e))
                    traceback.print_exc()
        print("Final Length of sources mac_roman: {}".format(len(sources)))
        return sources,headers

def get_gdf_buffer(
        data:pd.DataFrame,
        geog_crs:str = geog_crs_default,
        proj_crs:str = proj_crs_default
        ):
    ''' Take in a Pandas DataFrame and convert it to a GeoPandas GeoDataFrame by extracting
    and setting Plume Longitude/Plume Latitude for its geometry.
    Also create a shape (Pandas GeoSeries of Points) with a 150m buffer around them.
    Inputs:
        data (pd.DataFrame): source/candidate plume list
        geog_crs (string): plume list geographical coordinate reference system
        proj_crs (string): project coordinate reference system to reproject to for calculations

    Outputs:
        gdf (gp.GeoDataFrame): GeoPandas GeoDataFrame with geometry set to Plume Lon/Lat and reprojected to project CRS
        shp (gp.GeoDataFrame): GeoPandas GeoDataFrame containing only Source ID, Plume Lon/Lat, and uniqueID, without extra data, but with a 150m buffer.
    '''
    # convert to GeoDataFrame with lon/lat Coordinate Reference System
    data_clipped = data[['Source identifier','Plume Longitude (deg)','Plume Latitude (deg)','uniqueID']]
    geometry = [sp.geometry.Point(xy) for xy in zip(data['Plume Longitude (deg)'].astype('float'), data['Plume Latitude (deg)'].astype('float'))]
    gdf = gp.GeoDataFrame(data,crs=geog_crs_default,geometry=geometry)
    shp = gp.GeoDataFrame(data_clipped,crs=geog_crs_default,geometry=geometry)

    # reproject to project Coordinate Reference System
    gdf = gdf.to_crs(crs=proj_crs)
    shp = shp.to_crs(crs=proj_crs)

    # add 150m buffer to shp
    shp.geometry = shp.buffer(150)

    return gdf,shp

def calculate_overlap(gdf:gp.GeoDataFrame,shp:gp.GeoSeries):
    ''' Take in a GeoPandas GeoDataFrame of the candidate plume points and an equivalent GeoDataFrame with a 150m buffer.
    Returns a DataFrame containing percentage area overlap between the two.
    Inputs:
        gdf (gp.GeoDataFrame): GeoPandas GeoDataFrame with geometry set to Plume Lon/Lat and reprojected to project CRS
        shp (gp.GeoDataFrame): GeoPandas GeoDataFrame containing only Source ID, Plume Lon/Lat, and uniqueID, without extra data, but with a 150m buffer.

    Outputs:
        overlaps (pd.DataFrame): Pandas DataFrame containing Source IDs (SID) and corresponding percentage of area overlap.
    '''
    overlaps = dict()
    # source_ids = gdf['Source identifier'].tolist()
    source_ids = []
    pcts_overlap = []
    for index,row in gdf.iterrows():
        source_id = gdf.loc[index,'Source identifier']
        poly = shp.geometry[index]                              # polygon associated with source ID
        intersect_poly = shp[shp['geometry'].intersects(poly)]  # all polygons that intersect with poly
        poly2 = intersect_poly.drop([index])                    # exclude poly from list of overlapping polygons

        if (len(poly2) > 0):                                    # if there is overlap, calculate % overlap
            # print(source_id)
            # create gdf with area overlap
            poly_overlap = poly2.drop(columns=['geometry'])
            geometries = [poly.intersection(p['geometry']) for index,p in poly2.iterrows()]
            area_overlap = gp.GeoDataFrame(poly_overlap,geometry=geometries)

            # % overlap = area(overlap)/area(poly)
            overlap_area = area_overlap.geometry.area
            poly_area = poly.area
            pct_overlap = overlap_area/poly_area

            source_ids.append(source_id)
            pcts_overlap.append(pct_overlap.iloc[0])             # convert pd.Series to float
        else:
            source_ids.append(source_id)
            pcts_overlap.append(0)                               # no overlap

    overlaps['SID'] = source_ids
    overlaps['V2'] = pcts_overlap
    overlaps = pd.DataFrame(overlaps,index=gdf.index)

    return overlaps

def filter_plumes_recursive(overlaps:pd.DataFrame, data:gp.GeoDataFrame, flux_colname:[str], max_overlap:float=max_overlap_default):
    ''' Take in Pandas DataFrame of plumes with % overlap, GeoPandas GeoDataFrame of grouped data subset, list of flux column names,
    and maximum allowed overlap.
    Returns a GeoPandas DataFrame where plumes with duplicate overlap but lower flux or NaN values are filtered out.
    Inputs:
        overlaps (pd.DataFrame): Pandas DataFrame containing Source IDs (SID) and corresponding percentage of area overlap.
        data (gp.GeoDataFrame): GeoPandas GeoDataFrame with geometry set to Plume Lon/Lat and reprojected to project CRS
        flux_colname (string list): list of the DataFrame columns containing flux measurements
        max_overlap (float): maximum allowed overlap, decimal form

    Outputs:
        data (gp.GeoDataFrame): GeoPandas GeoDataFrame where plumes with duplicate overlap but lower flux or NaN values are filtered out.
    '''
    filtered_overlaps = overlaps[overlaps.V2 > max_overlap]
    nfiltered_overlaps = len(filtered_overlaps)
    # do nothing if
    # 1. 0 or 1 plumes
    # 2. no overlaps or no overlaps > max
    if len(data) <= 1 or nfiltered_overlaps == 0:
        return data
    # if 2 plumes, and overlap > max, remove plume with lower flux
    elif nfiltered_overlaps <= 2:
        # check if duplicate overlaps
        dups = overlaps[overlaps.duplicated(subset=['V2'])]                                                 # get duplicate overlaps

        # if duplicate % overlap
        if len(dups) > 0:
            # compare plume flux
            # copy flux columns into overlaps df
            for col in flux_colname:
                overlaps[col] = data[col]

            # pre-calculate min flux of each row for both overlaps and data
            overlaps['flux_min'] = overlaps[flux_colname].min(axis=1)                                       # pre-calculate min flux of each row
            data['flux_min'] = data[flux_colname].min(axis=1)                                               # pre-calculate min flux of each row

            # loop over dups and iteratively remove minimum flux
            for index,row in dups.iterrows():
                # for each row, get next row that has duplicate overlap
                source1 = row['SID']                                                                        # current row SID
                flux1 = overlaps.loc[index]                                                                 # current row in overlaps df
                flux2 = overlaps[overlaps['V2'] == flux1['V2']]                                             # get next row with same flux as current row
                flux2 = flux2[(flux2['SID'] != source1)]                                                    # remove source1 from list of duplicate overlaps
                source2 = flux2['SID'].iloc[0]                                                              # next row SID of duplicate overlap
                subset = overlaps[overlaps['SID'].isin([source1,source2])]

                # if one of the plumes only has nans, keep the other plume
                nans = overlaps[overlaps['flux_min'].apply(np.isnan)]
                if len(nans) > 0:
                    nans_sid = nans['SID'].iloc[0]
                    data = data[data['Source identifier'] != nans_sid]

                else:
                    min_flux = subset['flux_min'].min()                                                         # get lower flux value
                    data = data[data['flux_min'] != min_flux]                                                   # filter out data row that has duplicate overlap and lower flux

            data = data.drop(columns=['flux_min'])                                                              # drop 'flux_min' column from result
            return data

        # if no duplicate % overlap, just return plume with highest flux
        else:
            # easier to append fluxes to overlaps and work with that first
            for col in flux_colname:
                overlaps[col] = data[col]

            # get lower flux
            overlaps['flux_max'] = overlaps[flux_colname].max(axis=1)
            lower_flux = overlaps[flux_colname].max(axis=1).min()

            # filter out plume with lower flux and return
            data = data[overlaps['flux_max'] != lower_flux]
            return data

    # if more than 2 plumes, remove plume with highest % overlap
    else:
        # remove plume with highest % overlap
        max_overlap_plume_ind = overlaps['V2'].idxmax()                 # get index of plume with highest overlap
        data = data.drop([max_overlap_plume_ind])                        # drop plume with highest overlap from data
        overlaps = overlaps.drop([max_overlap_plume_ind])                # drop plume with highest overlap from overlaps

        # recalculate % overlap
        gdf,shp = get_gdf_buffer(data,geog_crs_default,proj_crs_default)
        overlaps = calculate_overlap(gdf,shp)

        # recurse, now with one less plume
        return filter_plumes_recursive(overlaps,data,flux_colname,max_overlap)

def filter_plumes(data:pd.DataFrame,flux_colname:[str]):
    ''' Take in a Pandas DataFrame of plumes grouped by uniqueID.
    Return a GeoPandas GeoDataFrame where plumes with overlap above max allowed overlap or NaN values are filtered out.
    Inputs:
        data (pd.DataFrame): Pandas DataFrame of plumes with the same uniqueID
        flux_colname (string list): list of the DataFrame columns containing flux measurements

    Outputs:
        data (gp.GeoDataFrame): GeoPandas GeoDataFrame where plumes with duplicate overlap but lower flux or NaN values are filtered out.
    '''
    # if data df has more than 1 row
    if data.shape[0] > 1:
        gdf,shp = get_gdf_buffer(data,geog_crs_default,proj_crs_default)
        overlaps = calculate_overlap(gdf,shp)
        filtered = filter_plumes_recursive(overlaps,gdf,flux_colname,max_overlap_default)
        data = filtered.drop(columns=['geometry'])                                              # returns a gdf; we want a normal df
        return data
    else:
        return data

def flux_overest(df_path:str, output_path:str, max_overlap:float, aws=False, target_bucket=None):
    ''' Filter out plumes within the same search radius with a pre-defined max radial overlap.
    Inputs:
        df_path (string): Path to initial plume list CSV file.
        outpath (string): Output filepath for filtered plume list CSV.
        max_overlap (float): Decimal percentage of maximum plume radial overlap.

    Returns:
        void
        The process produces and writes a CSV file to specified outpath.
    '''
    t1 = datetime.now()
    print(t1)
    print("flux_overest : df_path:{}, output_path:{}, max_overlap:{}, aws={}, target_bucket:{}".format(df_path, output_path, max_overlap, aws, target_bucket))
    sources, headers = read_plume_list(df_path)
    #print("\nreturn headers : \n{}".format(headers))
    print("return sources length: {}".format(len(sources)))

    # convert to dataframe
    dct = {}
    for i in range(len(headers)):
        try:
            dct[headers[i].strip()] = [source[i] for source in sources]
        except Exception as e:
            print("Error populating dct : {}".format(str(e)))
    print("dct length : {}".format(len(dct)))

    try:
        source_df = pd.DataFrame(dct)
    except Exceptions as e:
        print("Error in DataFrame : {}".format(str(e)))

    #print("source_df : {}".format(source_df))
    print("source_df done : {}".format(datetime.now()))
    print("source_df length : {}".format(len(source_df.index)))
    # create unique ID to sort by
    # "Nearest facility (best estimate)" + "Line name"
    if "# Line name" in source_df:
        line_name = source_df['# Line name']
    else:
        line_name = source_df['Line name']
    #print("line_name : {}".format(line_name))
    
    if 'Nearest facility (best estimate)' in source_df:
        nearest_facility = source_df['Nearest facility (best estimate)']
        uniqueID = [nearest_facility[i]+line_name[i] for i in range(len(nearest_facility))]
    else:
        uniqueID = line_name

    source_df['uniqueID'] = uniqueID
    source_df['Flight_Run'] = line_name

    print("{} unique_id and Flight run done".format(datetime.now()))

    # duplicate source IDs -> append unique Candidate ID letter to each source ID
    if 'Source ID' in source_df:
        source_id = source_df['Source ID']
    else:
        source_id = source_df['Source identifier']

    print("source_id done: {}".format(datetime.now()))
    candidate_id = source_df['Candidate ID']
    source_id_unique = [source_id[i]+candidate_id[i][-2:] for i in range(len(source_id))]
    source_df['Source identifier'] = source_id_unique

    print("{} Length of source_id_unique : {}".format(datetime.now(), len(source_id_unique)))

    # replace null values as NAN
    source_df = source_df.replace('#VALUE!',np.nan)

    # column names with flux
    flux_colname = [col for col in headers if 'm wind: E (kg/hr)' in col] + [col for col in headers if 'Emission Rate (kg/hr) [HRRR' in col]

    print("{} length of flux_colname : {}".format(datetime.now(), len(flux_colname)))
    # cast numerical columns needed to floats
    for col_name in flux_colname:
        source_df[col_name] = pd.to_numeric(source_df[col_name], downcast="float")

    # split dataframe by created uniqueID
    grouped = source_df.groupby('uniqueID')

    # create a new dataframe to aggregate filtered results
    filtered_df = pd.DataFrame({},columns=source_df.columns)
    print("{} sourcedf length : {}".format(datetime.now(), len(source_df)))
    #print(source_df.head())

    # process and filter plume list
    print("list(grouped.groups.keys() : {}".format(len(list(grouped.groups.keys()))))
    i=0
    for plume in list(grouped.groups.keys()):
        # plume = '4K Dairy Farm Partnershipang20170616t203426'
        # print(plume)
        data = grouped.get_group(plume)
        # print(data)
        # get plumes with flux (filter data where flux columns > 0)
        #    build condition
        #    e.g.: data.loc[data['10 m wind: E (kg/hr)'].astype('float64') > 1 | data['2 m wind: E (kg/hr)'].astype('float64') > 1]
        cond = 'data = data.loc['
        for col in flux_colname:
            cond = cond + ' (data[\'{}\'].astype(\'float64\') > 1) |'.format(col)
        cond = cond[:-2] + ' ]'

        # subset with condition
        exec(cond)
        i=i+1
        if i%50==0:
            print("condition execution completed : {}".format(i))

        

        #remove identical fluxes
        data = data.drop_duplicates(subset=flux_colname)

        # filter plumes
        filtered_data = filter_plumes(data,flux_colname)
        filtered_df = filtered_df.append(filtered_data)

    print("filtered data done : {}".format(datetime.now()))
    # cleanup and write to csv
    filtered_df = filtered_df.drop(columns=['uniqueID'])
    filtered_df['Source identifier'] = filtered_df['Source identifier'].apply(lambda s: s[:-2])
    print(len(filtered_df))
    # print(filtered_df.head())

    t2 = datetime.now()
    print(t2)
    print("TIME : {}".format(t2-t1))
    if aws:
        output_localp = '/tmp/'+output_path.split('/')[-1]
        print('writing output file to {}'.format(output_localp))
        with open(output_localp, 'w') as f:
            filtered_df.to_csv(path_or_buf=f,index=True)
            #source_df.to_csv(path_or_buf=f,index=True)
            print('updated dataframe written to csv')

        s3 = boto3.resource('s3')
        bucket = s3.Bucket(target_bucket)
        bucket.upload_file(output_localp,output_path)
        print('output csv {} uploaded to s3 {}'.format(output_localp,output_path))
    else:
        with open(output_path, 'w') as f:
            filtered_df.to_csv(path_or_buf=f,index=True)
        print('updated dataframe written to csv {}'.format(output_path))
    print("DONE : {}".format(datetime.now()))

def main(AWS=False, event={}):
    # default params
    plume_fname = ''
    out_fname = 'filter_out.csv'
    max_overlap = max_overlap_default

    if AWS:

        key = event['filename']
        plume_fname = '/tmp/'+key
        #out_fname = plume_fname.replace('plumes_ext', 'plumes_cluster')

        plume_dir = '/'.join(plume_fname.split('/')[:-1])
        if not os.path.exists(plume_dir):
            os.makedirs(plume_dir)

        # download the plume file
        s3 = boto3.resource('s3')
        bucket = s3.Bucket(event['bucket'])
        print('download {} to {}'.format(key, plume_fname))
        bucket.download_file(key, plume_fname)

   
        output_file_name=""
        if "input" in plume_fname.lower():
            output_file_name=plume_fname.lower().replace("input", "output")
        else:
            output_file_name = os.path.splitext(plume_fname)[0]+"_output.csv"

        output_path = 'data/plume_filtering/output/{}'.format(output_file_name.split('/')[-1])

        max_overlap = max_overlap_default
        print("plume_fname : {}, output_path : {}, max_overlap : {}".format(plume_fname, output_path, max_overlap))
        try:
            flux_overest(plume_fname, output_path, max_overlap_default, aws=True, target_bucket=event['bucket'])
        except Exception as e:
            print("ERROR IN flux_overest : {}".format(str(e)))
    else:
        plume_fname, output_path, max_overlap = parse_args()
        print("plume_fname : {}, output_path : {}, max_overlap : {}".format(plume_fname, output_path, max_overlap))
        try:
            flux_overest(plume_fname, output_path, max_overlap)
        except Exception as e:
            print("ERROR IN flux_overest : {}".format(str(e)))

def lambda_handler(event, context):
    bucket = event['Records'][0]["s3"]["bucket"]["name"]
    filename = event['Records'][0]["s3"]["object"]["key"]
    event = {'bucket':bucket, 'filename':filename}
    main(AWS=True, event=event)
    return {
        'statusCode': 200,
    }  

if __name__ == "__main__":
    main()
