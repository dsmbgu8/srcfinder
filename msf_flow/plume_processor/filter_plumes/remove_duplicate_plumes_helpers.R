#HELPER FUNCTIONS FOR remove_duplicate_plumes.R file


# MAKE.SHAPEFILE FUNCTION------------------------------------------------------
# Description: function to create a 150m buffer around plume points (lat/lon) 
#              from the manual plume/source list

# Necessary packages: library(rgdal), library(sp), library(rgeos), library(raster)

# Assumptions: the plume latitude longitude are in column 9 and 8, respectively, 
#              geog.crs is lat/long, proj.crs is in meters

# Inputs:
#     data - data frame with plume lat/long in columns 9 and 8, respectively 
#     geog.crs - CRS class of the geographic coordinate PROJ.4 projection system (lat/long)
#         ex// CRS('+proj=longlat +datum=WGS84 +no_defs +ellps=WGS84 +towgs84=0,0,0')
#     proj.crs - CRS class of the projected coordinate PROJ.4 projection system to use (MUST BE IN METERS)

# Output: SpatialPolygonsDataFrame class 

make.shapefile = function(data, geog.crs, proj.crs){
  shp = SpatialPointsDataFrame(data[,9:8], data=data, proj4string=geog.crs) # use plume lat/long to create pts
  shp = spTransform(shp, proj.crs) # change projection to meters
  shp = gBuffer(shp, byid=TRUE, width=150) # put a 150m buffer around the points (aka: search radius or max fetch)
  return(shp)
}




# POLY.OVERLAP FUNCTION--------------------------------------------------------
# Description: function to calculate percent overlap between two plumes with 
#              150m search buffer 

# Necessary packages: library(testit), library(raster)

# Assumptions: input shapefile has a "Source.identifer" column & shapefile 
#              does not have self-intersecting polygons

# Inputs:
#     poly1 - source identifer from manual source list 
#     poly2 - 2nd source identifer from manual source list 
#     shp - shapefile of plume points with 150m buffers (output from make.shapefile)
#          with attribute table including a column with source ID

# Output: numeric value of overlap

poly.overlap = function(poly1, poly2, shp){
  
  ### subset the shapefile to only include polygon with specified source ID
  poly1 = subset(shp, shp@data$Source.identifier == poly1) 
  poly2 = subset(shp, shp@data$Source.identifier == poly2)
  
  ### if polygons don't overlap ==> % overlap =0
  if (has_warning(intersect(poly1, poly2)) == TRUE) {
    answer=0  # only have warning when two polygons do not overlap
  
  ### calculate % overlap
  } else{
    return(area(intersect(poly1, poly2))/area(poly1)*100)
  }
}




# PERCENT.OVERLAP FUNCTION-----------------------------------------------------
# Description: function to generate table with source ID and total percent 
#             overlap for each plume

# Necessary packages: library(testit)

# Internal function calls: poly.overlap

# Assumptions: input shapefile has a "Source.identifer" column and shapefile 
#             does not have self-intersecting polygons

# Inputs:
#     shp - shapefile of points with 150m buffer (output from make.shapefile)
#     data - plume list as a csv

# Output: data frame with two columns: source ID and percent overlap

percent.overlap = function(shp, data){
  
  ### create df 
  per.overlap = data.frame(SID = 1)
  
  ### add source ID and percent overlap to table
  for (i in 1:nrow(data)){

    per.overlap[i,1] = data$Source.identifier[i] # set first column to be source ID
    
    poly = subset(shp, shp@data$Source.identifier == data$Source.identifier[i]) # get polygon associated with source ID
    poly_intersect = crop(shp, poly) # get all polygons that intersect poly (polygon of interest)
    
    ### get only overlapping polygons w/o including poly
    poly2 = subset(poly_intersect, poly_intersect@data$Source.identifier != data$Source.identifier[i]) 
    
    ### if there is overlap between poly and other polygons ==> calculate % overlap
    if (nrow(poly2@data) != 0){ 
      print(data$Source.identifier[i])
      area_overlap = crop(poly, poly2)  # get the area of overlap 
      per.overlap[i, 2] = area(area_overlap)/area(poly) # calculate %overlap (area overlap/area of poly)
    
    } else{per.overlap[i, 2]=0} # no overlap
  }
  
  return(per.overlap)

}




# RECURSIVE.OVERLAP FUNCTION---------------------------------------------------
# Description: function to remove sources from a df that have less than user 
#             specified overlap

# Necessary packages: library(rgdal), library(sp), library(rgeos), library(raster), 
#                     library(testit)

# Internal function calls: make.shapefile() and percent.overlap()

# Assumptions: input data has "Source.identifer" & "X2.m.wind..E..kg.hr." col names

# Inputs:
#     test - data frame with two columns - source ID and percent overlap 
#            (output from percent.overlap() function)
#     data - plume/source list as a R data.frame 
#     max.over - numeric of max allowable overlap between two polygons (should be a decimal) 

# Output: input data variable modified to have duplicate sources removed

recursive.overlap = function(test, data, max.over){
  
  ### if max overlap is less than user defined number, then we're done
  if (max(as.numeric(test[,2])) <= max.over){ # check if max % overlap in test df is <= user defined max overlap
    return(data)
  
  }else{
    
    if(length(unique(test[,2])) <= 2){ 
      
      ### if <= 2 plumes: keep data if only one plume or remove plume with lower flux 
      if (nrow(data) == 1){ # check if only 1 row in data df
        return(data)
      
      ### if two plumes have the same % overlap, remove the plume with lower flux 
      } else if (length(subset(test, duplicated(test[,2]))[,2]) > 0){ #check if any duplicate % overlap values in test df
        
        dups = subset(test, duplicated(test[,2])) #create a df that only has duplicate % overlap source ID
        
        for (i in 1:nrow(dups)){
          flux1 = subset(data, data$Source.identifier == dups[i,1]) #get the row from data df that is duplicated
          source2 = subset(test, test[,2] == dups[i,2] & test[,1] != dups[i,1]) #get 2nd source ID with same overlap as flux1
          flux2 = subset(data, data$Source.identifier == source2[1,1]) #get 2nd row from data df that is duplicated
          fluxmin = min(as.numeric(flux1$X2.m.wind..E..kg.hr.), as.numeric(flux2$X2.m.wind..E..kg.hr.), na.rm=T)
          data = subset(data, data$X2.m.wind..E..kg.hr. != fluxmin) # subset data df to exclude min flux
        } 
        
        return(data)
        
      } else if (max(as.numeric(test[,2]), na.rm=T) <= max.over){ # may not need this?
        return(data)
        
      } else{
        return(subset(data, data$X2.m.wind..E..kg.hr.== max(as.numeric(data$X2.m.wind..E..kg.hr., na.rm=T))))
      
      }
      
    ### if there are more than two plumes, remove the plume with the highest percent overlap  
    } else{
      
      ### remove plume with highest percent overlap
      dff = subset(test, test[,2] == max(as.numeric(test[,2])), na.rm=T) #find plume with highest percent overlap
      data = subset(data, data$Source.identifier != dff[1,1]) # remove that plume from the data df
      test = subset(test, test[,2] != max(as.numeric(test[,2]), na.rm=T)) #remove plume with highest % overlap from test df
      
      ### recalculate percent overlap 
      shp = make.shapefile(data, geog.crs, proj.crs) # regenerate search radii
      test = percent.overlap(shp, data) # recalculate percent overlap
      
      ### repeat until max overlap between plumes is less than user specified amount
      return(recursive.overlap(test, data, max.over))
    }
  }
}

