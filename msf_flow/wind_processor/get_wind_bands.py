import sys
import re
from os.path import basename, splitext
from osgeo import gdal
from osgeo import ogr
from osgeo import osr
#from osgeo import gdal_array
#from osgeo import gdalconst
import numpy as np
import argparse
import logging
from math import log10
from netCDF4 import Dataset
from datetime import datetime, timezone

def init_logger(logger):
    """Initialize logger.
    """
    log_level = logging.WARNING
    #log_level = logging.INFO
    logger.setLevel(log_level)
    ch = logging.StreamHandler()
    ch.setLevel(log_level)
    logger.addHandler(ch)

def parse_args():
    """Retrieve command line parameters.
    
    Returns:
        ArgumentParse: command line parameters
    """
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("-i", "--infile", required=True,
                        help="path to input file")
    parser.add_argument("-o", "--outfile", required=False,
                        help="path to output NetCDF file")
    args = parser.parse_args()
    in_fname = args.infile
    in_fname_base = basename(in_fname)
    if args.outfile:
        out_fname = args.outfile
    else:
        out_fname = splitext(in_fname_base)[0].\
                    replace(".", "_").replace(" ", "_") + ".nc"
    return in_fname, in_fname_base, out_fname

def pixel2coord(x, y, gt):
     xoff, a, b, yoff, d, e = gt
     xp = gt[1] * x + gt[2] * y + gt[0]
     yp = gt[4] * x + gt[5] * y + gt[3]
     return(xp, yp)

def extract_bands(src_ds, band_nums, dst_fname, dst_driver, band_names=None,
                  xsize=None, ysize=None, gt=None, wkt=None):
    # If any of the optional parameters giving information about the source
    # dataset are not provided, get them from the source dataset.
    if xsize is None:
        xsize = src_ds.RasterXSize
    if ysize is None:
        ysize = src_ds.RasterYSize
    if gt is None:
        gt = src_ds.GetGeoTransform()
    if wkt is None:
        wkt = src_ds.GetProjection()
    dst_nbands = len(band_nums)
    dst_ds = dst_driver.Create(dst_fname, xsize=xsize, ysize=ysize,
                               bands=dst_nbands,
                               eType=gdal.GDT_Float32)
    dst_ds.SetGeoTransform(gt)
    dst_ds.SetProjection(wkt)
    dst_band_num = 1 # GDAL counts band numbers starting at 1.
    for src_band_num in band_nums:
        band = src_ds.GetRasterBand(src_band_num)
        out_band = dst_ds.GetRasterBand(dst_band_num)
        out_band.WriteArray(band.ReadAsArray())
        dst_band_num += 1

    dst_ds.FlushCache()
    return dst_ds

def reproject(src_ds, dst_spatial_ref, dst_fname, dst_driver, dst_coord_res,
              band_names=None, src_xsize=None, src_ysize=None, src_nbands=None,
              src_gt=None, src_wkt=None, src_spatial_ref=None,
              fill=None, logger=None):
    # If any of the optional parameters giving information about the source
    # dataset are not provided, get them from the source dataset.
    if src_xsize is None:
        src_xsize = src_ds.RasterXSize
    if src_ysize is None:
        src_ysize = src_ds.RasterYSize
    if src_nbands is None:
        src_nbands = src_ds.RasterCount
    if src_gt is None:
        src_gt = src_ds.GetGeoTransform()
    if src_wkt is None:
        src_wkt = src_ds.GetProjection()
    if src_spatial_ref is None:
        src_spatial_ref = osr.SpatialReference(wkt=src_wkt)
        
    # Construct the coordinate transformation
    ct = osr.CoordinateTransformation(src_spatial_ref, dst_spatial_ref)

    # Determine geometric transform, width and height of the projection
    src_coords = np.array([[pixel2coord(x, y, src_gt)
                            for x in range(src_xsize)]
                           for y in range(src_ysize)])
    dst_coords = np.array([[ct.TransformPoint(src_coords[y,x,0],
                                              src_coords[y,x,1])
                            for x in range(src_xsize)]
                           for y in range(src_ysize)])
    dst_x = dst_coords[:,:,0]
    dst_y = dst_coords[:,:,1]
    num_decimals = -int(log10(dst_coord_res))
    min_dst_x_rnd = round(np.min(dst_x), num_decimals)
    min_dst_y_rnd = round(np.min(dst_y), num_decimals)
    max_dst_x_rnd = round(np.max(dst_x), num_decimals)
    max_dst_y_rnd = round(np.max(dst_y), num_decimals)
    dst_height = int((max_dst_y_rnd - min_dst_y_rnd) / dst_coord_res + 0.5) + 1
    dst_width = int((max_dst_x_rnd - min_dst_x_rnd) / dst_coord_res + 0.5) + 1
    dst_gt = (min_dst_x_rnd, dst_coord_res, 0.0,
              max_dst_y_rnd, 0.0, -dst_coord_res)
    logger.info("Reprojected size is height={} x width={}".format(dst_height,
                                                                  dst_width))

    # Create destination dataset
    dst_ds = dst_driver.Create(dst_fname, xsize=dst_width, ysize=dst_height,
                               bands=src_nbands, eType=gdal.GDT_Float32)
    dst_wkt = dst_spatial_ref.ExportToWkt()
    dst_ds.SetGeoTransform(dst_gt)
    dst_ds.SetProjection(dst_wkt)

    # Set fill value in all bands
    if fill is not None:
        for i in range(src_nbands):
            band = dst_ds.GetRasterBand(i+1)
            band.SetNoDataValue(fill)
            band.Fill(fill, 0.0)

    # Reproject from the source to the destination coordinates.
    res = gdal.ReprojectImage(src_ds, dst_ds, src_wkt, dst_wkt,
                              gdal.GRA_Bilinear)
    dst_ds.FlushCache()
    return dst_ds

def get_global_meta(out_fname, time_utc,
                    min_x, max_x, min_y, max_y):
    utcnow = datetime.utcnow()
    today = utcnow.strftime("%Y%m%d")
    global_meta = {"title": "HRRR hourly winds",
                   "summary": "HRRR hourly winds reprojected to WGS-84 lat-lon",
                   "institution": "Jet Propulsion Laboratory, California Institute of Technology",
                   "creator_name": "Jet Propulsion Laboratory, California Institute of Technology",
                   "creator_url": "http://www.jpl.nasa.gov",
                   "keywords": "Atmospheric modeling, Meteorological factors, Geospatial analysis",
                   "keywords_vocabulary": "2017 IEEE Taxonomy Version 1.0",
                   "standard_name_vocabulary": "CF Standard Names v67",
                   "Conventions": "CF-1.7, ACDD-1.3",
                   "cdm_data_type": "Image",
                   "date_created": "{}".format(today),
                   "date_modified": "{}".format(today),
                   "date_issued": "{}".format(today),
                   "id": "{}".format(splitext(out_fname)[0]),
                   "naming_authority": "JPL CEDAS Project",
                   "comment": "",
                   "processing_level": "L3",
                   "time_coverage_start": "{}".format(time_utc),
                   "time_coverage_end": "{}".format(time_utc),
                   "geospatial_lat_min": "{:.3f}".format(min_y),
                   "geospatial_lat_max": "{:.3f}".format(max_y),
                   "geospatial_lat_units": "degrees_north",
                   "geospatial_lon_min": "{:.3f}".format(min_x),
                   "geospatial_lon_max": "{:.3f}".format(max_x),
                   "geospatial_lon_units": "degrees_east",
                   "geospatial_vertical_min": "10",
                   "geospatial_vertical_max": "80",
                   "geospatial_vertical_units": "meters above ground",
                   "geospatial_vertical_positive": "up",
                   "source": "HRRR Archive at the University of Utah (https://doi.org/10.7278/S5JQ0Z5B/)",
                   "history": "{}: Wind speed bands extracted and reprojected to WGS-84 lat-lon coordinates".format(today),
                   "references": "",
                   "acknowledgement": "This data was produced at the Jet Propulsion Laboratory, California Institute of Technology, under contract with the National Aeronautics and Space Administration. This work was sponsored by JPL's Center for Data Science and Technology.",
                   "agency": "NASA",
                   "center": "JPL",
                   "project": "CEDAS"
    }
    return global_meta

def write_nc(ds, time_val, time_utc=None, out_fname="out.nc", x_var_name="x", x_meta=None,
             y_var_name="y", y_meta=None, t_var_name="time", t_meta=None,
             default_var_name="val", out_var_names=None, out_var_meta=None,
             fill=None, xsize=None, ysize=None, gt=None, logger=None):
    # Get input data, size and geometric transform
    data = ds.ReadAsArray()
    data_shape = data.shape
    if time_utc is None:
        time_utc = datetime.fromtimestamp(time_val).strftime("%Y%m%d")
    if out_var_names is None:
        out_var_names = [default_var_name for i in range(data_shape[0])]
    assert (data_shape[0] == len(out_var_names)), "Number of variable names in out_var_names does not match number of output data bands!"
    if out_var_meta is not None:
        assert (data_shape[0] == len(out_var_meta)), "Number of metadata dictionaries in out_var_meta does not match number of output data bands!"
    if xsize is None:
        xsize = ds.RasterXSize
    if ysize is None:
        ysize = ds.RasterYSize
    if gt is None:
        gt = ds.GetGeoTransform()

    # Calculate spatial coordinates
    coords = np.array([[pixel2coord(x, y, gt)
                        for x in range(xsize)]
                       for y in range(ysize)])
    x_vals = np.array([pixel2coord(x, 0, gt)[0] for x in range(xsize)])
    y_vals = np.array([pixel2coord(0, y, gt)[1] for y in range(ysize)])

    # Create NetCDF dataset, dimensions, and independent variables
    rootgrp = Dataset(out_fname, "w", format="NETCDF4")
    rootgrp.createDimension(x_var_name, data_shape[2])
    rootgrp.createDimension(y_var_name, data_shape[1])
    rootgrp.createDimension(t_var_name, 1)
    x = rootgrp.createVariable(x_var_name, "f4", dimensions=(x_var_name,),
                               zlib=True)
    x[:] = x_vals
    if x_meta is not None:
        for key,val in x_meta.items():
            x.setncattr(key, val)
    y = rootgrp.createVariable(y_var_name, "f4", dimensions=(y_var_name,),
                               zlib=True)
    y[:] = y_vals
    if y_meta is not None:
        for key,val in y_meta.items():
            y.setncattr(key, val)
    t = rootgrp.createVariable(t_var_name, "u8", dimensions=(t_var_name,),
                               zlib=True)
    t[0] = time_val
    if t_meta is not None:
        for key,val in t_meta.items():
            t.setncattr(key, val)

    # Set global attributes
    min_x = np.min(x_vals)
    max_x = np.max(x_vals)
    min_y = np.min(y_vals)
    max_y = np.max(y_vals)
    global_meta = get_global_meta(out_fname, time_utc,
                                  min_x, max_x, min_y, max_y)
    for key,val in global_meta.items():
        rootgrp.setncattr(key, val)
    
    # Create variable and attributes for each band
    for band_num in range(data_shape[0]):
        data_band = data[band_num,:,:]
        vals = rootgrp.createVariable(out_var_names[band_num], data_band.dtype,
                                      dimensions=(t_var_name, y_var_name,
                                                  x_var_name,),
                                      fill_value=fill, zlib=True)
        vals[:,:] = data_band
        if out_var_meta is not None:
            for key,val in out_var_meta[band_num].items():
                vals.setncattr(key, val)
    rootgrp.close()
    if logger is not None:
        logger.critical("Selected bands reprojected to WGS-84 Lat/Lon coordinates written to {}".format(out_fname))
    
def time_from_fname(fname, logger=None):
    regex = re.compile('(\d{4}\d*)')
    match = regex.search(fname)
    if match is None:
        if logger is not None:
            logger.error("Could not extract date from filename")
        sys.exit(1)
    elif len(match.groups()) != 1:
        if logger is not None:
            logger.error("More than one substring in filename matches date regex")
        sys.exit(2)
    else:
        date_str = match[1]
        date_str_len = len(date_str)
        if date_str_len == 4:
            date_fmt = '%Y'
        elif date_str_len == 6:
            date_fmt = '%Y%m'
        if date_str_len == 4:
            date_fmt = '%Y'
        elif date_str_len == 8:
            date_fmt = '%Y%m%d'
        elif date_str_len == 10:
            date_fmt = '%Y%m%d%H'
        elif date_str_len == 12:
            date_fmt = '%Y%m%d%H%M'
        elif date_str_len == 14:
            date_fmt = '%Y%m%d%H%M%S'
        else:
            if logger is not None:
                logger.error("Date substring {} found in file doesn't match a known format".format(date_str))
            sys.exit(3)
        gran_date = datetime.strptime(date_str, date_fmt)
        gran_date_utc = datetime(gran_date.year, gran_date.month,
                                 gran_date.day, gran_date.hour,
                                 gran_date.minute, gran_date.second,
                                 tzinfo=timezone.utc)
        return gran_date_utc

def main():
    """Main program.  Parse arguments, and read winds from the specified 
    granule.
    """
    # Initializer logger
    logger = logging.getLogger(__name__)
    init_logger(logger)

    # Parse input file name from command-line arguments
    in_fname, in_fname_base, out_fname = parse_args()
    in_fname_base_parts = splitext(in_fname_base)
    subset_fname = in_fname_base_parts[0] + "_subset" + ".nc"
    HRRR, RTMA = 0, 1
    if "hrrr" in in_fname_base_parts[0]:
        data_source = HRRR
    elif "rtma" in in_fname_base_parts[0]:
        data_source = RTMA
    # Get time value from filename
    time_utc = time_from_fname(in_fname_base, logger=logger)
    time_val = int(time_utc.timestamp())

    # Open input file and retrieve its raster and coordinate system and
    # projection information
    src_ds = gdal.Open(in_fname)
    logger.info(src_ds.GetMetadata())
    src_nbands = src_ds.RasterCount
    logger.info('Got {} bands'.format(src_nbands))
    src_xsize = src_ds.RasterXSize
    src_ysize = src_ds.RasterYSize
    logger.info('Raster ysize x xsize = {} X {}'.format(src_ysize, src_xsize))
    src_gt = src_ds.GetGeoTransform()
    logger.info(src_gt)
    src_wkt = src_ds.GetProjection()
    logger.info(src_wkt)
    src_spatial_ref = osr.SpatialReference(wkt=src_wkt)

    # Create dataset with just the necessary wind-related bands from the
    # original data.
    fill = -9999.9
    if data_source == HRRR:
        # HRRR settings
        band_nums = [8, 55, 56, 71, 72, 73]
        htgl = [0, 80, 80, 10, 10, 10]
        band_names = ["wind_speed_gust",
                      "eastward_wind_80m",
                      "northward_wind_80m",
                      "eastward_wind_10m",
                      "northward_wind_10m",
                      "wind_speed_10m"]
        band_long_names = ["wind_speed_of_gust_at_surface",
                           "eastward_wind_at_80m_above_ground",
                           "northward_wind_at_80m_above_ground",
                           "eastward_wind_at_10m_above_ground",
                           "northward_wind_at_10m_above_ground",
                           "wind_speed_at_10m_above_ground"]
    elif data_source == RTMA:
        # RTMA settings
        band_nums = [5, 6, 8, 9, 10]
        htgl = [10, 10, 10, 10, 10]
        band_names = ["eastward_wind_10m",
                      "northward_wind_10m",
                      "wind_from_direction_10m",
                      "wind_speed_10m",
                      "wind_speed_gust"]
        band_long_names = ["eastward_wind_at_10m_above_ground",
                           "northward_wind_at_10m_above_ground",
                           "wind_from_direction_at_10m_above_ground",
                           "wind_speed_at_10m_above_ground",
                           "wind_speed_of_gust_at_10m_above_ground"]
    band_meta = [{"long_name": band_long_names[i],
                  "standard_name": band_long_names[i],
                  "units": "m/s",
                  "source": "reprojected from {}".format(in_fname_base),
                  "coverage_content_type": "physicalMeasurement",
                  "height": htgl[i],
                  "height_units": "meters above ground",
                  "band_number_in_original_granule": band_nums[i]}
                 for i in range(len(band_nums))]
    src_subset_ds = extract_bands(src_ds, band_nums, "",
                                  gdal.GetDriverByName('MEM'),
                                  band_names=band_names,
                                  xsize=src_xsize, ysize=src_ysize,
                                  gt=src_gt, wkt=src_wkt)
    gdal.GetDriverByName("NetCDF").CreateCopy(subset_fname, src_subset_ds)
    logger.critical("Wrote subset wind-related data to {}".format(subset_fname))

    # Create dataset with the selected bands reprojected to
    # regular WGS-84 lat/lon coordinates.  We transform each input coordinate
    # to lat-lon, and take the minimum and maximum coordinates to determine
    # the bounds for the destination geometric transform.  If we had examined
    # just the corners, there is potential clipping.  We pick a latitude
    # and longitude resolution that produces a similar size number of grid
    # cells as in the original granule.
    dst_spatial_ref = osr.SpatialReference()
    dst_spatial_ref.ImportFromEPSG(4326) # ESPG:4326 = WGS-84 Lat-Lon
    dst_coord_res = 0.025
    dst_ds = reproject(src_subset_ds, dst_spatial_ref,
                       "", gdal.GetDriverByName('MEM'),
                       dst_coord_res, src_xsize=src_xsize, src_ysize=src_ysize,
                       src_nbands=len(band_nums),
                       src_gt=src_gt, src_wkt=src_wkt,
                       src_spatial_ref=src_spatial_ref,
                       fill=fill, logger=logger)

    # Save output dataset as NetCDF
    lon_var_name = "lon"
    lon_meta = {"long_name": "longitude",
                "standard_name": "longitude",
                "units": "degrees_east",
                "source": "reprojected from {}".format(in_fname_base),
                "coverage_content_type": "coordinate"}
    lat_var_name = "lat"
    lat_meta = {"long_name": "latitude",
                "standard_name": "latitude",
                "units": "degrees_north",
                "source": "reprojected from {}".format(in_fname_base),
                "coverage_content_type": "coordinate"}
    time_var_name = "time"
    time_meta = {"long_name": "time_utc",
                 "standard_name": "time",
                 "units": "seconds since 1970-1-1 0:0:0 UTC",
                 "coverage_content_type": "coordinate"}

    write_nc(dst_ds, time_val, time_utc=time_utc, out_fname=out_fname,
             x_var_name=lon_var_name, x_meta=lon_meta,
             y_var_name=lat_var_name, y_meta=lat_meta,
             t_var_name=time_var_name, t_meta=time_meta,
             out_var_names=band_names, out_var_meta=band_meta,
             fill=fill, logger=logger)

    # Close GDAL datasets
    src_ds = None
    src_subset_ds = None 
    dst_ds = None


if __name__ == "__main__":
    main()
