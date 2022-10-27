####################################################################
#
#                 calculating source persistence
# -----------------------------------------------------------------
#
#     Author:  Kelsey Foster
#
#     This script was written to calculate source persistence.
#
#####################################################################

# load necessary R packages----------------------------------------
library(rgdal)
library(sp)
library(raster)
library(plyr)

# USER DEFINED VARIABLES---------------------------------------------
flightlines = '/Users/eyam/M2AF/MSF_manual_data_workflow/scripts/2_source_persistence_calculation/source_persistence/flightlines_all_20181105.shp' # fall 2016 and 2017 flightlines footprints
source.list = '/Users/eyam/M2AF/MSF_manual_data_workflow/scripts/2_source_persistence_calculation/source_persistence/plume-list-20190413.csv' # source list
outpath = '/Users/eyam/M2AF/MSF_manual_data_workflow/scripts/2_source_persistence_calculation/source_persistence/' 
outfile = 'source_persistence_output_v20190415.csv'

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




source.persistence = function(source.list, flightlines, outpath, outfile){
  
    # load the data into R
    flightlines = shapefile(flightlines)
    source.list=read.csv(source.list)
    
    # create source points
    geog.crs = CRS('+proj=longlat +datum=WGS84 +no_defs +ellps=WGS84 +towgs84=0,0,0')
    source.pts = subset(source.list, !duplicated(source.list$Source.identifier)) # get unique values for source ID
    source.pts = SpatialPointsDataFrame(source.pts[,3:2], data=source.pts, proj4string=geog.crs) # turn unique IDs into a shapefile
    
    # standardized source ID column name
    colnames(source.list)[1] = 'source.ID'
    colnames(source.pts@data)[1] = 'source.ID'
    
    ### 1. count the number of plumes detected at each source
    plume_persistence = plyr::count(source.list, vars='source.ID')
    colnames(plume_persistence)[colnames(plume_persistence)=="freq"] <- "observed.plumes" # rename columns

  
    ### 2. find the intersection between sources and flightlines
    source.intersect = raster::intersect(source.pts, flightlines)
    print(source.intersect)


    ### 3. count the number of flightlines that intersect each source
    intersect.count = plyr::count(source.intersect@data, vars='source.ID')
    colnames(intersect.count)[colnames(intersect.count)=="freq"] <- "total.overflights" # rename columns
    
    
    ### 4. calculate persistence of CH4 detection: total CH4 observations from plume list/total flightlines
    
    # combine datasets into attribute table of source.pts
    source.pts@data = merge(source.pts@data, plume_persistence, all=TRUE, by='source.ID')
    source.pts@data = merge(source.pts@data, intersect.count, all=TRUE, by='source.ID')
    
    # calculate persistence of CH4 detection
    source.pts@data$Source.persistence = eval(parse(text= 'observed.plumes / total.overflights'), source.pts@data)
    
    # export attribute table data as csv
    file.path = paste0(outpath, outfile)
    write.csv(source.pts@data, file = file.path, row.names=F)
    print('check outpath')
}


source.persistence(source.list, flightlines, outpath, outfile)

