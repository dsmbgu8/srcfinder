#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 14 10:12:53 2019

@author: vineet
"""
import numpy as np
import pygrib 
import csv
import requests
from datetime import datetime
from datetime import timedelta
import sys
import itertools

import os
AWS = 'AWS' in os.environ.keys() and os.environ['AWS'] == 'TRUE'
if AWS:
    import boto3

def distanceSpherical(x, y, distanceType='haversine', logger=None):
    """ %DISTANCE_ Determine distances between locations
    
    This function produces a matrix that describes the
    distances between to sets of locations.
    
    INPUT PARAMETERS
       x - location coordinates for data set #1 [n1 x D]
       y - location coordinates for data set #2 [n2 x D]
    OUTPUT PARAMETERS
       h - distance between points in x from points in y [n1 x n2]
    
    EXAMPLE:
       x=[0,0; 5,0; 5,5]
       y=[1,1; 5,5]
       h = distance_(x,y)
    RESULT:
        1.4142    7.0711
        4.1231    5.0000
        5.6569         0
    
    EXTERNAL FUNCTIONS CALLED: none
    WRITEN BY: VINEET YADAV"""
    
    x = np.asmatrix(x)
    y = np.asmatrix(y)
    x_rows,x_cols = x.shape
    y_rows,y_cols = y.shape
    # Parameters
    if x_cols != y_cols :
        if logger is not None:
            logger.error('ERROR in DISTANCE_: locations must have same number of dimensions (columns)')
        return -1
    h = np.zeros((x_rows,y_rows))
    if distanceType == 'euclid' :
        for i in range(0, x_cols) :
            multPart1 = x[:,i] @ np.ones((1, y_rows))
            multPart2 = np.ones((x_rows,1)) @ np.transpose(y[:,i])
            h = h + np.power((multPart1 - multPart2),2)
        h = np.sqrt(h)
        return(h) # Returned Distance is in Kilometers
    elif distanceType == 'sphericalCosines' : # Spherical law of Cosines
        earthRadius = 6378.137 # WGS 84 Equatorial Radius
        x = x * (np.pi / 180)
        y = y * (np.pi / 180)
        multPart1 = np.sin(x[:,1] @ np.ones((1, y_rows)))
        multPart2 = np.sin(np.ones((x_rows,1)) @ np.transpose(y[:,1]))
        multPart3 = np.cos(x[:,1] @ np.ones((1, y_rows)))
        multPart4 = np.cos(np.ones((x_rows,1)) @ np.transpose(y[:,1]))
        multPart5 = np.cos(x[:,0] @ np.ones((1, y_rows)) - \
                              np.ones((x_rows,1)) @ np.transpose(y[:,0]))
        h = earthRadius * np.arccos(np.multiply(multPart1,multPart2) + \
                                     np.multiply(np.multiply(multPart3, multPart4),multPart5))
        h = np.real(h)
    elif distanceType == 'haversine' :
        polarRadius = 6356.7523
        equatorRadius = 6378.137
        meanRadius = (2 * equatorRadius + polarRadius) / 3
        if x_rows == y_rows :
            # Longitude Part
            multPart1 = x[:,0] @ np.ones((1, y_rows))
            multPart2 = np.ones((x_rows,1)) @ np.transpose(y[:,0])
            dlon = (multPart2 - multPart1) * (np.pi / 180)
            # latitude Part
            multPart1 = x[:,1] @ np.ones((1, y_rows))
            multPart2 = np.ones((x_rows,1)) @ np.transpose(y[:,1])
            dlat = (multPart1 - multPart2) * (np.pi / 180)
            latRadians = multPart1 * (np.pi / 180)
            A = np.power(np.sin(dlat / 2),2)
            B = np.multiply(np.cos(latRadians),np.cos(np.transpose(latRadians)))
            C = np.power(np.sin(np.transpose(dlon) / 2),2)
            dtemp = A + np.multiply(B,C)
            h = meanRadius * 2 * np.arctan2(np.sqrt(dtemp),np.sqrt(1 - dtemp))
        else :
            if (x_rows > y_rows):
                x,y = y,x
                x_rows,x_cols = x.shape
                y_rows,y_cols = y.shape
                state = True
            else :
                state = False
            h = np.zeros((x_rows,y_rows))
            latY = np.multiply(y[:,1],(np.pi / 180))
            for i in range(0,x_rows):
                dlat1 = x[i,1] - y[:,1]
                dlon1 = x[i,0] - y[:,0]
                dlat = dlat1 * (np.pi / 180)
                dlon = dlon1 * (np.pi / 180)
                latRad = x[i,1] * (np.pi / 180)
                A = np.power(np.sin(dlat / 2),2)
                B = np.multiply(np.cos(latRad),np.cos(latY[i]))
                C = np.power(np.sin(dlon / 2),2)
                dtemp = A + np.multiply(B,C)
                h[i,:] = np.transpose(meanRadius * 2 * np.arctan2(np.sqrt(dtemp),np.sqrt(1 - dtemp)))
            if (state):
                h = np.transpose(h)  
    return(np.asarray(h)) # Returned Distance is in Kilometers

# Return minimum and maximum distance of point to matrix of latlon with the
# index of the point
def distanceIndex(x, y, indxType='min', dtype='euclid', logger=None):
    """Get indices and distance of the nearest point to the supplied point x
    from a list of coordinates in y"""
    # Note Indices Returned are Zero Based
    if logger is not None:
        logger.info("x={}".format(x))
        logger.info("y={}".format(y))
    x = np.asmatrix(x)
    y = np.asmatrix(y)
    x_rows,x_cols = x.shape
    y_rows,y_cols = y.shape
    # Parameters
    if x_cols != y_cols :
        if logger is not None:
            logger.error('ERROR in DISTANCE_: locations must have same number of dimensions (columns)')
        return -1
    h = np.zeros((x_rows,y_rows))
    earthRadius = 6378.137 # WGS 84 Equatorial Radius
    x = x * (np.pi / 180) # Convert to Radians
    y = y * (np.pi / 180)
    if dtype == 'spherical':
        multPart1 = np.sin(x[:,1] @ np.ones((1, y_rows)))
        multPart2 = np.sin(np.ones((x_rows,1)) @ np.transpose(y[:,1]))
        multPart3 = np.cos(x[:,1] @ np.ones((1, y_rows)))
        multPart4 = np.cos(np.ones((x_rows,1)) @ np.transpose(y[:,1]))
        multPart5 = np.cos(x[:,0] @ np.ones((1, y_rows)) - \
                          np.ones((x_rows,1)) @ np.transpose(y[:,0]))
        h = earthRadius * np.arccos(np.multiply(multPart1,multPart2) + \
                                 np.multiply(np.multiply(multPart3, multPart4),multPart5))
        h = np.real(h)
        h = np.transpose(h)
    elif dtype == 'euclid':
        h = np.sqrt(np.square(x[:,0] - y[:,0]) + np.square(x[:,1] - y[:,1]))
        
    indices = np.arange(0,y_rows) # 0 based indexing
    indices = np.reshape(indices,(y_rows,1))
    distIndex = np.concatenate((indices,h),axis=1)
    distIndex = np.asarray(distIndex)
    distIndex = distIndex[np.argsort(distIndex[:,1])]
    if indxType == 'min' :
        mnIndx = distIndex[0,0]
        mndist = distIndex[0,1]
        return (mnIndx, mndist, distIndex)
    elif indxType == 'max' :
        mxIndx = distIndex[y_rows - 1,0]
        mxdist = distIndex[y_rows - 1,1]
        return (mxIndx, mxdist, distIndex)    
    
def windMNSTD(npoints, plumeloc, fileArgv, alt=10, logger=None):
    """ Four different types of Mean and Std of Wind Speed are returned
    for winds at the given altitude.  An exception is raised if no wind band 
    is found at that altitude.
    1. if nearest points (npoints=1) desired is 1 and number of files supplied 
     is 1 then wind speed of nearest point is returned and population
     standard deviation is 0
    2. if nearest points (npoints=1) desired are 1 and number of files supplied
    is > 1 then mean wind speed of nearest points to the plume location in 
    all files is returned and similarly population standard deviation reported 
    is the std of the nearest points in all the supplied files
    3. if nearest points (npoints>1) desired are > 1 and number of files =1
    then mean and std of npoints surrounding the plume location for the file
    supplied is returned.
    4. if nearest points (npoints>1) desired are >1 and number of files >1
    then mean and std of all points across all files is reported
    NOTE Overall in the output
    (1) first output is overall mean wind speed
    (2) second outpur is overall population standard deviation
    (3) mean and population standard devaition of npoints surrounding plume location
    (4) min location index and 
    (5) min distance to nearest point
    within each file is reported
    NOTE 2: This distance computation should work for both HRRR and RTMA
    data products and files can also be mixed thus first file can be RTMA
    and second file can be HRRR file and so on and so forth.  Note that
    HRRR has winds at 10 m and 80 m altitudes and RTMA has only 10 m.
    plumeloc is plume [lon lat] as numpy matrix or 1by2 numpy array and fileArgv
    is a list of files across which windspeed would be computed"""
    
    if logger is not None:
        logger.info("npoints={}, plumeloc={}, fileArgv={}".
                    format(npoints, plumeloc, fileArgv))
    tfiles = len(fileArgv) 
    # Assign Space for Storing Output
    windCompute = np.zeros((tfiles,2))
    grandMeanStd = np.matrix(np.zeros((npoints,tfiles)))
    filename_insideDate=[0]*tfiles
    # Compute Mean and Std of Wind Speed
    counter = 0

    # Oddly, the 10 m wind bands have the altitude in the band name, but
    # the 80 m wind bands do not.
    if alt == 10:
        # Valid for both HRRR and RTMA
        u_band_name = '10 metre U wind component'
        v_band_name = '10 metre V wind component'
    elif alt == 80:
        # Only valid for HRRR
        u_band_name = 'U component of wind'
        v_band_name = 'V component of wind'
    else:
        raise ValueError("Altitude {} not valid for recognized data types".format(alt))
    gust_band_name = 'Wind speed (gust)'

    # download files and convert to local paths if on AWS S3
    fileArgv_local = fileArgv
    # download only relevant wind data files from S3 if on AWS
    if AWS and tfiles > 0:
        # s3://wind-bucket/path/to/wind/data -> wind-bucket
        wind_bucket = fileArgv[0].split('s3://')[1].split('/')[0]

        # convert filenames to local paths
        # s3://wind-bucket/path/to/wind/data -> /tmp/path/to/wind/data
        fileArgv_local = ['/tmp'+fname.split('s3://'+wind_bucket)[1]
            for fname in fileArgv]

        # create wind local path if it doesn't exist
        # s3://wind-bucket/path/to/wind/data -> path/to/wind
        wind_local_path = '/'.join(fileArgv_local[0].split('/')[:-1])
        print('wind local path is {}'.format(wind_local_path))
        if not os.path.isdir(wind_local_path):
            print('creating local dir {}'.format(wind_local_path))
            os.makedirs(wind_local_path)

        bucket = boto3.resource('s3').Bucket(wind_bucket)
        for fname_local in fileArgv_local:
            if not os.path.isfile(fname_local):
                fname_cloud = fname_local.split('/tmp/')[1]
                print('downloading s3://{}/{} to {}'.format(wind_bucket,fname_cloud,fname_local))
                bucket.download_file(fname_cloud,fname_local)
            else:
                print('{} already downloaded'.format(fname_local))

    # Process each file.
    for i in range(0,tfiles):
        file = fileArgv_local[i]
        if logger is not None:
            logger.info("Working on {}".format(file))
        grbs = pygrib.open(file)
        if logger is not None:
            for grb in grbs:
                logger.info(grb)
            grbs.rewind
        try:
            grb = grbs.select(name=u_band_name)[0]
        except:
            raise KeyError("Band {} not found in {}".format(u_band_name, file))
        wind10U = grb.values
        lat,lon = grb.latlons() # Note as the grid is same we only have to do it once
        tm=grb.validDate
        YYYYMMDDHHMM = [tm.year,tm.month,tm.day,tm.hour,tm.minute,tm.second,
                        tm.microsecond]
        filename_insideDate[i]=[file,YYYYMMDDHHMM]
        del grb
        # U10 wind
        try:
            grb = grbs.select(name=v_band_name)[0]
        except:
            raise KeyError("Band {} not found in {}".format(v_band_name, file))
        wind10V = grb.values
        del grb
        # Wind Gust
        try:
            grb = grbs.select(name=gust_band_name)[0]
        except:
            raise KeyError("Band {} not found in {}".format(gust_band_name,
                                                            file))
        windGust = grb.values
        del grb
        vecCoord = np.concatenate((np.matrix(np.ravel(lon,order='F')), 
                           np.matrix(np.ravel(lat,order='F'))))
        vecCoord = np.transpose(vecCoord)
        # Get Sorted Distances
        # mode = 'euclid' # Faster, but less accurate than "spherical"
        mode = 'spherical'
        mnIndx, mndist, index = distanceIndex(plumeloc,vecCoord, 'min', mode,
                                              logger=logger)
        # Vectorize wind10U, wind10v and Windgust
        windData = np.concatenate((np.matrix(np.ravel(wind10U,order='F')), 
                           np.matrix(np.ravel(wind10V,order='F')),
                           np.matrix(np.ravel(windGust,order='F'))))
        # windData column order
        windData = np.transpose(windData)
        # Formula to get wind speed
        windData[:,[0,1]] = np.square(windData[:,[0,1]]) # Square of
        # U and V component of wind in column 1 and 2
        # Take Sqrt to get Wind Speed
        windSpeed = np.sqrt(windData[:,0] + windData[:,1])
        # Conditions for getting mean wind speed and standard deviation
        if npoints == 1 :
           meanWindSpeed = np.mean(windSpeed[int(mnIndx)])
           # Standard Deviation of a single point is zero
           stdWindSpeed = np.std(windSpeed[int(mnIndx)])
        elif npoints > 1 :
           # Mean for selected points in each file
           extrctIndex=index[:,0].astype(int)
           meanWindSpeed = np.mean(windSpeed[extrctIndex[0:npoints]]) 
           stdWindSpeed = np.std(windSpeed[extrctIndex[0:npoints]])
           grandMeanStd[:,i] = windSpeed[extrctIndex[0:npoints]]
        windCompute[counter,:] = np.matrix([meanWindSpeed,stdWindSpeed])
        grbs.close()
        counter+=1
    grandMeanStd=np.ravel(grandMeanStd,order='F')
    if npoints == 1 and tfiles == 1 : 
        return(meanWindSpeed,0,mnIndx,mndist,windCompute,filename_insideDate)
    elif npoints > 1 and tfiles == 1 :
        return(meanWindSpeed,stdWindSpeed,mnIndx,mndist,windCompute,filename_insideDate)
    elif npoints > 1 and tfiles > 1 :
        return(np.mean(grandMeanStd),np.std(grandMeanStd),
               mnIndx,mndist,windCompute,filename_insideDate)
    elif npoints == 1 and tfiles > 1 :
        return(np.mean(grandMeanStd),np.std(grandMeanStd),mnIndx,
               mndist,windCompute,filename_insideDate)
      
def plumetimeFormat(utcTime,utcFormat):
    """The purpose of this function is to take UTC time given in format
    8/21/2018 18:49:59 UTC and extract data, year, month, hour, minute,second
    to report as a string as integertime and as strings in format YYMMDDHHMMSS.
    utcformat can only be utcstring or numericstring in case of utcstring it can
    only be in form 8/21/2018 18:49:59 UTC. The function returns as a list
    containing
    [year,month,day,hour,minute,second,microsecond]. This is the first output
    second output is the same time in format YYMMDDHHMMSS. It only takes one string
    at a time.

    """
    case_1='utcString'
    case_2='numericString'
    
    if utcFormat.lower() == case_1.lower():
        timeSplit = utcTime.split(('/'))
        withinDaySplit = utcTime.split((':'))
        hSplit = withinDaySplit[0].split((' '))
        
        tm = datetime(int(timeSplit[2][0:4]), # year
                               int(timeSplit[0]), # month
                               int(timeSplit[1]), # day
                               int(hSplit[1]), # hour
                               int(withinDaySplit[1]), # minute
                               int(withinDaySplit[2][0:2])) # second
    if utcFormat.lower() == case_2.lower():

        tm = datetime(int(utcTime[0:4]), # year
                               int(utcTime[4:6]), # month
                               int(utcTime[6:8]), # day
                               int(utcTime[8:10]), # hour
                               int(utcTime[10:12]), # minute
                               int(utcTime[12:14])) # second
        
    intTime = [tm.year,tm.month,tm.day,tm.hour,tm.minute,tm.second,tm.microsecond]
    YYYYMMDDHHMMSS = tm.strftime("%Y%m%d%H%M%S")  
    return(intTime,YYYYMMDDHHMMSS)
        
def nearstHM(utcTime,boundingHours,bounding15Minutes,utcFormat):
    """
    The main purpose of the function is to return both numeric bounding file 
    time periods for a given plume time. The function return these time strings
    as a list one for HRRR files which are top of the hour and RRTMA files
    that are rounded up to closest 15 minutes. 
    INPUTS
        (1) utCTime can be in this form 8/21/2018 18:49:59 UTC in which case
        utcFormat should be utcstring. The second format in which time can be
        given is YYYYMMDDHHMMSS in which case utcFormat should be numericstring
        (2) boundingHours (integer) For hrrr files it gives boundinghours 
        for the current hour in which the plume time lies. This input takes only 
        integers. Thus if it is -1 then assumes that the filename is not desired 
        and it resturns an empty list. If a 0 input is supplied then it returns 
        a numeric string just specifying the current hour in which plume lies. 
        In Case it is >0 then it gives list of bounding hours surrounding current hour. 
        Negative argument is useful if only numeric string for RRTMA is desired
        (3) bounding15Minutes (integer) Similar two second argument but for RRTMA 
        files with rounding to 15 minutes based on closeness of plume time to 
        nearest 15 minutes. It returns the string as a list. Again negative number 
        returns an empty list, a value of 0 returns the closest 15 minutes and a 
        positive number gives surrounding 15 minutes
        (4) utcFormat (string) input can only be numericstring or utcstring see text for
        input 1.
    OUTPUT
       (1) lists of bounding time for hrrr (hrfile) and bounding minutes for
       RRTMA time (minutefilenames)
    """
    case_1='numericString'
    case_2='utcString'
    # Check for utcformat string. Case does not matter
    if (utcFormat.lower() != case_1.lower() and utcFormat.lower() != case_2.lower()):
        if logger is not None:
            logger.error('ERROS IN SUPPLIED UTCFORMAT:FAILURE')
        return -1

    intTime, _ = plumetimeFormat(utcTime,utcFormat) # Get time in a numeric list
    # Extract year,month,day,hour,minute,second
    year_ = intTime[0]
    month_ = intTime[1]
    day_ = intTime[2]
    hour_ = intTime[3]
    minute_ = intTime[4]
    second_ = intTime[5]
   
    if boundingHours > 0: 
        givenTimeHr = datetime(year_,month_,day_,hour_)
        spacedNumbers = np.linspace(1,2 * boundingHours + 1,2 * boundingHours + 1)
        medianHour = np.median(spacedNumbers)
        # Assign space for list to store file time strings
        hourFile = [0] * (2 * boundingHours + 1) 
        counter = 0
        for i in range(1,2 * boundingHours + 2):
            if i < medianHour:
                # Get timestring below median time
                tm = givenTimeHr - timedelta(hours=i)
                hourFile[counter] = tm.strftime("%Y%m%d%H%M")
                counter+=1
                # Get timestring above median time
                tm = givenTimeHr + timedelta(hours=i)
                hourFile[counter] = tm.strftime("%Y%m%d%H%M")
                counter+=1
            elif i == medianHour:
                # Get timestring for median hour or
                # UTC hour for which wind speed is desired
                tm = givenTimeHr
                hourFile[counter] = tm.strftime("%Y%m%d%H%M")
                counter+=1
    # If input is 0 then return hour string in which the current time lies            
    elif boundingHours == 0:
        givenTimeHr = datetime(year_,month_,day_,hour_)
        hourFile = [0]
        tm = givenTimeHr
        hourFile[0] = tm.strftime("%Y%m%d%H%M")
    # If input is <0 then return empty list    
    elif boundingHours < 0:
        hourFile = []
        
    if bounding15Minutes > 0:    
        spacedNumbers = np.linspace(1,2 * bounding15Minutes + 1,
                                      2 * bounding15Minutes + 1)
        medianMn = np.median(spacedNumbers)
        givenTimeMn = datetime(year_,month_,day_,hour_,
                                          minute_,second_)
        minuteFilenames = [0] * (2 * bounding15Minutes + 1)
    
        counter = 0
        for i in range(1,2 * bounding15Minutes + 2):
        
            if i < medianMn:
                # Get timestring below median time
                tm = givenTimeMn - timedelta(minutes=i * 15)
                tm += timedelta(minutes=7.5)
                tm -= timedelta(minutes=tm.minute % 15,
                                         seconds=tm.second,
                                         microseconds=tm.microsecond)
                minuteFilenames[counter] = tm.strftime("%Y%m%d%H%M")
                counter+=1
                # Get timestring above median time
                tm = givenTimeMn + timedelta(minutes=i * 15)
                tm += timedelta(minutes=7.5)
                tm -= timedelta(minutes=tm.minute % 15,
                      seconds=tm.second,microseconds=tm.microsecond)
                minuteFilenames[counter] = tm.strftime("%Y%m%d%H%M")
                counter+=1
                
            elif i == medianMn:
                # Get time string for median hour
                tm = givenTimeMn
                tm += timedelta(minutes=7.5)
                tm -= timedelta(minutes=tm.minute % 15,
                                         seconds=tm.second,
                                         microseconds=tm.microsecond)
                minuteFilenames[counter] = tm.strftime("%Y%m%d%H%M")
                counter+=1
    elif bounding15Minutes == 0:
        givenTimeMn = datetime(year_,month_,day_,hour_,
                                          minute_,second_)
        minuteFilenames = [0]
        tm = givenTimeMn
        tm += timedelta(minutes=7.5)
        tm -= timedelta(minutes=tm.minute % 15,
                                 seconds=tm.second,
                                 microseconds=tm.microsecond)
        minuteFilenames[0] = tm.strftime("%Y%m%d%H%M")
        
    elif bounding15Minutes < 0:
        minuteFilenames = []
        
         
    return(hourFile, minuteFilenames)

def readPlumes(filename, logger=None):
    """
    read plumes from filename that contains plume time and lat lon
    """
    if logger is not None:
        logger.info("reading {}".format(filename))
    with open(filename,'rt') as fin:
        plumes = list(csv.DictReader(fin, skipinitialspace=True))
    return plumes
    
def gatherPlumes(filelist, logger=None):
    """
    Gather plume time and lat lon from multiple files or a single file
    """
    # This function gathers list of plumes and their lat lon. It returns
    # them as a list of tuples with time, lat and lon entries.
    # The input filelist should be a list
    # containing filenames with their path. These files can reside in
    # one directory or multiple directories; the code only uses files
    # listed as an input to the function to gather plumes.

    # Important things to Note:
    # (1) Note if the field names in the header of plume files change
    # (that is it is not Candidata ID and Lat Lon) then please modify
    # function readPlumes.
    # (2) Time is in format YYYYmmddHHMMSS as a number
    # in a first column in the numpy array.

    plumes = list(itertools.chain.from_iterable([readPlumes(fname,
                                                            logger=logger)
                                                 for fname in filelist]))
    return(plumes)

# %% Get Data From Nearest Stations To Plume Based on Mesowest API   
def stationWindSpeed(lon, lat, beginHour, endHour, searchRadius=10,
                     token=None, tout=60, sType='any',
                     variables="wind_speed,wind_gust,wind_direction",
                     logger=None):
    
    """ Gather windspeed data from nearest weather stations or nearest national
    weather stations based on a radius surrounding the plume lat lon. This uses
    restful API of mesowest to gather the data"""
    # lon = longitude around which station data would be searched
    # lat = latitude around which station data would be searched
    # beginHour = beginning temporal window for getting the data
    # endHour = end temporal window for getting the data
    # searchRadius = in km within given lon lat around which stations would be
    # searched
    # token = authntication token from mesowest api
    # tout = timeout for the server in seconds
    # on return
    # windspeed_=average windspeed from the nearest station
    # min_dist = distance to the nearest station
    # stationData=data from all the stations surrounding given lat lon
    
    lat = str(lat)
    lon = str(lon)
    searchRadius = str(searchRadius)
    beginHour = str(beginHour)
    endHour = str(endHour)
    try:
        if  sType == 'any':
            http_1 = 'https://api.mesowest.net/v2/stations/'
            http_2 = 'statistics?&radius=' + lat + ',' + lon + ',' + searchRadius + '&'
            http_3 = 'vars=' + variables + '&'
            http_4 = 'type=all&start=' + beginHour + '&end=' + endHour + '&'
            http_5 = 'type=all&token=' + token
            URL_ = http_1 + http_2 + http_3 + http_4 + http_5
        elif sType == 'nws':
            http_1 = 'https://api.mesowest.net/v2/stations/'
            http_2 = 'statistics?&radius=' + lat + ',' + lon + ',' + searchRadius + '&'
            http_3 = 'vars=' + variables + '&'
            http_4 = 'type=all&start=' + beginHour + '&end=' + endHour + '&network=1&'
            http_5 = 'type=all&token=' + token
            URL_ = http_1 + http_2 + http_3 + http_4 + http_5
        if logger is not None:
            logger.info(URL_)
        response = requests.get(URL_,timeout=tout)
        if (response.status_code != 200):
            if logger is not None:
                logger.error('Bad Status Code')
            return(float('NaN'), float('NaN'), {})
        elif (response.status_code == 200):
            stationData = response.json()
            objectCount = stationData['SUMMARY']['NUMBER_OF_OBJECTS']
            if objectCount > 0:
                dist = np.zeros((objectCount,3))
                for i in range(0,objectCount):
                    dist[i,0] = i
                    dist[i,1] = stationData['STATION'][i]['DISTANCE']
                    if len(stationData['STATION'][i]['STATISTICS']) != 0 and \
                    len(stationData['STATION'][i]['STATISTICS']['wind_speed_set_1']) != 0:
                        dist[i,2] = stationData['STATION'][i]['STATISTICS']['wind_speed_set_1']['average']
                    else:
                        dist[i,2] = np.NaN
                dist = dist[~np.isnan(dist).any(axis=1)]
                rows,_ = dist.shape
                if logger is not None:
                    logger.info("dist={}".format(dist))
                if rows != 0:
                    mnIndex = np.argsort(dist[:,1])
                    mnIndex = dist[mnIndex]
                    mnIndex = int(mnIndex[0,0])
                    windSpeed_ = stationData['STATION'][mnIndex]['STATISTICS']['wind_speed_set_1']['average']
                    min_dist = dist[mnIndex][1]
                    if logger is not None:
                        logger.info("windSpeed={}".format(windSpeed_))
                    return(windSpeed_, min_dist, stationData)
                else:
                    return(float('NaN'), float('NaN'), {})
            else:
                return(float('NaN'), float('NaN'), {})
    except:
        return(float('NaN'), float('NaN'), {})
