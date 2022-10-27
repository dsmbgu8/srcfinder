import glob
import os
import argparse
from math import sqrt
from collections import OrderedDict
from wind_processor import windspeed
from wind_processor.wind_type import WindType
from utils.logger import init_logger

def parse_args():
    """Retrieve command line parameters.
    
    Returns:
        ArgumentParse: command line parameters
    """
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("-p", "--plume_files", required=True,
                        help="path (with wildcards) to input plume files")
    parser.add_argument("-w", "--windir", required=True,
                        help="path to input wind file directory")
    parser.add_argument("-a", "--alt", required=False, type=int, default=10,
                        help="wind altitude")
    args = parser.parse_args()
    return args.plume_files, args.windir, args.alt

def get_mean_wind_key(wind_type, wind_alt, npoints, ntimes):
    mean_key = "Wind Mean (m/s) [{} {} m, {} nearest points for each of {} closest times]".format(wind_type, wind_alt, npoints, ntimes)
    std_key = "Wind Std (m/s) [{} {} m, {} nearest points for each of {} closest times]".format(wind_type, wind_alt, npoints, ntimes)
    return mean_key

def get_std_wind_key(wind_type, wind_alt, npoints, ntimes):
    std_key = "Wind Std (m/s) [{} {} m, {} nearest points for each of {} closest times]".format(wind_type, wind_alt, npoints, ntimes)
    return std_key

def compute_emission_rate(plume, wind_type, fill=None, default_fill="NA",
                          wind_alt=10, wind_ntimes=3, wind_npoints=10,
                          min_aspect_ratio=0.02, max_aspect_ratio=1.,
                          avg_ime_div_fetch20_key="AvgIMEdivFetch20 (kg/m)",
                          std_ime_div_fetch20_key="StdIMEdivFetch20 (kg/m)",
                          aspect_ratio_key="Aspect ratio20",
                          aspect_ratio_flag_key="Aspect Ratio Flag (0=valid, 1=invalid)",
                          emission_generic_key="Emission Rate (kg/hr)",
                          emission_uncertainty_generic_key="Emission Uncertainty (kg/hr)",
                          logger=None):
    if fill is not None:
        fill = str(fill)

    # Keys for wind mean and standard deviation.
    mean_wind_key = get_mean_wind_key(wind_type, wind_alt,
                                      wind_npoints, wind_ntimes)
    std_wind_key = get_std_wind_key(wind_type, wind_alt,
                                    wind_npoints, wind_ntimes)

    # Keys for output emission rate and uncertainty.
    emission_rate_key = "{} [{} {} m]".format(emission_generic_key,
                                             wind_type, wind_alt)
    emission_uncertainty_key = "{} [{} {} m]".\
        format(emission_uncertainty_generic_key, wind_type, wind_alt)

    # Calculate aspect ratio flag or set it to the fill value if the
    # inputs are not set.
    if ((aspect_ratio_key not in plume) or
        ((fill is not None) and (plume[aspect_ratio_key] == fill))):
        aspect_ratio_flag = default_fill if fill is None else fill
    else:
        aspect_ratio = float(plume[aspect_ratio_key])
        aspect_ratio_flag = int((aspect_ratio > max_aspect_ratio) or
                                (aspect_ratio < min_aspect_ratio))

    # Calculate emission rate or set it to the fill value if the
    # inputs are not set.
    if ((avg_ime_div_fetch20_key not in plume) or
        (mean_wind_key not in plume) or
        ((fill is not None) and
         ((plume[avg_ime_div_fetch20_key] == fill) or
          (plume[mean_wind_key] == fill)))):
        emission_rate = default_fill if fill is None else fill
    else:
        mean_wind = float(plume[mean_wind_key])
        if logger is not None:
            logger.info("Got mean wind {} for key {}".
                        format(mean_wind, mean_wind_key))
        avg_ime_div_fetch20 = float(plume[avg_ime_div_fetch20_key])
        emission_rate = avg_ime_div_fetch20 * mean_wind * 3600

    # Calculate emission rate uncertainty or set it to the fill value if the
    # inputs are not set.
    if ((emission_rate == fill) or
        (std_ime_div_fetch20_key not in plume) or
        (std_wind_key not in plume) or
        ((fill is not None) and
         ((plume[std_ime_div_fetch20_key] == fill) or
          (plume[std_wind_key] == fill)))):
        emission_uncertainty = default_fill if fill is None else fill
    else:
        std_wind = float(plume[std_wind_key])
        if logger is not None:
            logger.info("Got std wind {} for key {}".
                        format(std_wind, std_wind_key))
        std_ime_div_fetch20 = float(plume[std_ime_div_fetch20_key])
        if avg_ime_div_fetch20 < 1E-7:
            plume_std_div_mean = 0.
        else:
            plume_std_div_mean = std_ime_div_fetch20 / avg_ime_div_fetch20
        if mean_wind < 1E-7:
            wind_std_div_mean = 0.
        else:
            wind_std_div_mean = std_wind / mean_wind
        emission_uncertainty = sqrt(plume_std_div_mean *
                                    plume_std_div_mean +
                                    wind_std_div_mean *
                                    wind_std_div_mean) * emission_rate

    # Create dictionary of emission statistics.
    emission_stats = OrderedDict()
    emission_stats[aspect_ratio_flag_key] = aspect_ratio_flag
    emission_stats[emission_rate_key] = emission_rate
    emission_stats[emission_uncertainty_key] = emission_uncertainty
    return emission_stats

def compute_wind_stats(plume, winds_dir, wind_type=None, wind_alt=10,
                       fill=None, default_fill="NA", ntimes=3, npoints=10,
                       lat_key="Plume Latitude (deg)",
                       lon_key="Plume Longitude (deg)",
                       cand_id_key="Candidate ID", logger=None):
    if logger is not None:
        logger.info("ntimes={}, npoints={}".format(ntimes, npoints))

    if fill is not None:
        fill = str(fill)

    if wind_type is None:
        # Get wind type from path.
        wt = WindType(winds_dir)
        is_hrrr = wt.is_hrrr()
        is_rtma = wt.is_rtma()
        wind_type = wt.type_as_str()
    else:
        # Get wind type from function parameter.
        is_hrrr = (wind_type.lower() == "hrrr")
        is_rtma = (wind_type.lower() == "rtma")

    # We only support HRRR and RTMA for now.
    if not (is_hrrr or is_rtma):
        raise ValueError("Wind directory name must contain either \"hrrr\" or \"rtma\" (case-insensitive)")

    mean_wind_key = get_mean_wind_key(wind_type, wind_alt,
                                      npoints, ntimes)
    std_wind_key = get_std_wind_key(wind_type, wind_alt,
                                    npoints, ntimes)
    wind_stats = OrderedDict()

    if ((cand_id_key not in plume) or
        (lat_key not in plume) or
        (lon_key not in plume) or
        ((fill is not None) and
         ((plume[cand_id_key] == fill) or
          (plume[lat_key] == fill) or
          (plume[lon_key] == fill)))):
        wind_stats[mean_wind_key] = default_fill if fill is None else fill
        wind_stats[std_wind_key] = default_fill if fill is None else fill
    else:
        stringTime = plume[cand_id_key][3:11] + plume[cand_id_key][12:18]

        # Note this nearstHM function has many options
        # The function returns list of hourfile and minute file
        # hrfile is for HRRR and MN file is for RRTMA
        bounding = int(ntimes / 2)
        if is_hrrr:
            hrfiles, _ = windspeed.nearstHM(stringTime, bounding, -1,
                                            "numericstring")
            if logger is not None:
                logger.info("hrfiles={}".format(hrfiles))
            fllist = [os.path.join(winds_dir, hrfile[:8],
                                   "hrrr." + hrfile[:10] + \
                                   ".wrfsfcf00.grib2")
                      for hrfile in hrfiles]
        else:
            _, mnfiles = windspeed.nearstHM(stringTime, -1, bounding,
                                            "numericstring")
            if logger is not None:
                logger.info("mnfiles={}".format(mnfiles))
            fllist = [os.path.join(winds_dir, mnfile[:8],
                                   "rtma2p5_ru." + mnfile + \
                                   "z.2dvaranl_ndfd.grib2")
                      for mnfile in mnfiles]
        if logger is not None:
            logger.info("fllist={}".format(fllist))
        plume_coords = (float(plume[lon_key]), float(plume[lat_key]))
        cur_stats = windspeed.windMNSTD(npoints, plume_coords, fllist,
                                        alt=wind_alt, logger=logger)
        if logger is not None:
            logger.info("{}: {} file {} point wind stats for plume at {} = {}".\
                        format(wind_type, ntimes, npoints, plume_coords,
                               cur_stats))
        wind_stats[mean_wind_key] = cur_stats[0]
        wind_stats[std_wind_key] = cur_stats[1]
    return wind_stats

def main():
    # Initializer logger
    logger = init_logger()

    plume_files, winds_dir, wind_alt = parse_args()

    # This gets list of plume text files by wildcards in a directory. 
    listd = glob.glob(plume_files)
    if logger is not None:
        logger.info("listd={}".format(listd))

    # Gather list of plumes and their time and lat lon.
    plumes = windspeed.gatherPlumes(listd, logger=logger)
    if logger is not None:
        logger.info("plumes={}".format(plumes))
    wt = WindType(winds_dir)
    wind_type = wt.type_as_str()
    for plume in plumes:
        wind_stats = compute_wind_stats(plume, winds_dir, wind_type=wind_type,
                                        wind_alt=wind_alt, logger=logger)
        plume.update(wind_stats)
        emission_stats = compute_emission_rate(plume, wind_type, logger=logger)
        if logger is not None:
            logger.critical("output {}, {}".format(wind_stats, emission_stats))

if __name__ == "__main__":
    main()
