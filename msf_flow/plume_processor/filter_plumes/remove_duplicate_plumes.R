####################################################################
#
#      removing duplicate/overlapping plumes from plume list 
# -----------------------------------------------------------------
#
#     Author:  Kelsey Foster
#
#     This script was written to eliminate the possibility of 
#     double counting emissions from methane plumes within in  
#     the same search radius (maximum fetch = 150m). Please see
#     CA's Methane Super-emitters (Duren etal) supplementary
#     material (SI S2.5 and S2.8) for additional information.
#
#####################################################################

# load necessary R packages----------------------------------------
library(rgdal) 
library (sp) # gBuffer, SpatialPointsDataFrame, crs, and spTransform functions
library(rgeos) # gBuffer 
library(raster)
library(testit)


# USER DEFINED VARIABLES---------------------------------------------
df = read.csv('/Users/eyam/M2AF/MSF_manual_data_workflow/scripts/1_plume_proximity_filtering/flux_ovest_input_data.csv', stringsAsFactors = FALSE) #plume list as csv
out.file = "/Users/eyam/M2AF/MSF_manual_data_workflow/scripts/1_plume_proximity_filtering/flux_overest_output_06102019.csv" #output file
max.over = .30 #max allowable overlap between plume search radii (should be decimal)


# FUNCTIONS--------------------------------------------------------

# load helper functions
source('/Users/eyam/M2AF/MSF_manual_data_workflow/scripts/1_plume_proximity_filtering/remove_duplicate_plumes_helpers.R') 

### coordinate reference systems needed
geog.crs = CRS('+proj=longlat +datum=WGS84 +no_defs +ellps=WGS84 +towgs84=0,0,0') #make shapefile lat/long

proj.crs = CRS("+proj=aea +lat_1=34 +lat_2=40.5 +lat_0=0 +lon_0=-120 +x_0=0 +y_0=-4000000 
               +ellps=GRS80 +datum=NAD83 +units=m +no_defs") #change to m for area calculation (CA teale albers)

# main function
flux_overest = function(df, out.file, max.over){

  ### concatenate nearest facility and line name to get unique ID to sort by
  df$concat = paste0(df$Nearest.facility..best.estimate., df$Line.name) 

  ### There are duplicate source IDs ==> add unique candidate ID letter to end of each source ID
  df$Source.identifier = paste0(df$Source.identifier, substr(df$Candidate.ID, 19,20))
  
  ### split the df by concat
  x = split(df, df$concat) 

#   for (plume in 1:length(x)){
  for (plume in 1:25){
    print('')
    data = x[[plume]] # subset df by concat (nearest facility and flight line)
    data[data =='#VALUE!'] = NA # change null value to NA
    data = subset(data, data$X2.m.wind..E..kg.hr.>1) # get plumes with flux 
    data = subset(data, !duplicated(data$X2.m.wind..E..kg.hr.)) #remove identical fluxes
  
    if (nrow(data)>1){ #check if data df has more than 1 row
      print(plume)
  
      shp = make.shapefile(data, geog.crs, proj.crs)  # turn data df into a shapefile of points with 150m buffer
  
      per.overlap = percent.overlap(shp, data) #calculate %overlap and output a df with source ID and % overlap
  
      x[[plume]] = recursive.overlap(per.overlap, data, max.over) #remove all sources with > user defined amount of overlap
  
  
    }else{x[[plume]] = data}
  } 
  
  ### clean df and write to csv
  x = do.call('rbind', x) # recombine split df
  x$Source.identifier = substr(x$Source.identifier, 1,6) # remove candidate ID that was appended to source ID
  row.names(x) = NULL
  print(paste0('Check ', out.file, " for output table" ))
  write.csv(x, out.file, row.names=FALSE) # write df to a csv
  
}
  

flux_overest(df, out.file, max.over)


