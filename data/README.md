### Orthorectified CH4 CMF images 

Path: `/localstore/ang/y[yy]/cmf/ch4/ort` 

Each orthorectified CH4 Columnwise Matched Filter (CMF) image is a georeferenced ENVI/IDL image with 4 float32 bands. The first 3 bands are radiance bands at RGB wavelengths pulled directly from the corresponding L1 radiance data, and the 4th band is the CMF that gives pixelwise CH4 enhancements in units of parts-per-million / meter (ppmm). The current pipeline is based on unimodal CMF products which model the background distribution for each detector in the FPA as a unimodal Gaussian distribution. CMF image products use the product id 

`"angyy[yy]mmddtHHMMSS_ch4mf_v[calid]_img"` 

(e.g., `ang20200906t195820_ch4mf_v2y1_img`) where `[yy]` gives the 2-digit year indicating when the CMF was captured and `[calid]` is the version id specifying which settings were used in radiometric calibration for the CMF. We use the abbreviation `[cmf_img]` to refer to an arbitrary CMF image product below.

CMF NODATA pixels (w/ value -9999) either occur outside the boundaries of the captured scene extent or were not flagged as valid science pixels, and should be ignored. 
	
### CMF Label Images + Candidate/ROI Metadata 

Path: `/localstore/ang/y[yy]/cmf/ch4/ort/labels/latest`

We generate "weakly labeled" [[1]](#foot1) plume images for QCed flightlines by intersecting the (center of mass) latitude and longitude coordinates of expert QCed plumes with the connected components of their corresponding CMF CH4 enhancements above a user-specified minimum ppmm threshold.[[2]](#foot2). Each label image contains at most three possible rgb values: 

- positive "plume" pixels (rgb=<span style="color:#FFFFFF; background-color:red">red=[255,0,0]</span>)
- negative "false enhancement" pixels (rgb=<span style="background-color:#00FFFF">cyan=[0,255,255]</span>) 
- unlabeled pixels (rgb=<span style="color:#FFFFFF; background-color:#000000">black=[0,0,0]</span>). 

Label images use their associated CMF product id as their file prefix with suffix `"_mask.png"` (e.g., `ang20200906t195820_ch4mf_v2y1_img_mask.png`). 
	
We also generate several metadata products to preserve the mapping between the CMF label images computed from the candidate locations in the plume list and the metadata associated with each candidate. 

- `[cmf_img]_cid_meta.json`: mapping from candidate ids to plume list metadata. All metadata specified in plume list is captured here, in addition to details computed during the label generation processfor each candidate (e.g., label ROI bounding box, class label, area + major/minor axis lengths). 

- `[cmf_img]_roilab.tif`: single channel int32 image coregistered with CMF image containing pixelwise ROI labels for all candidates.

- `[cmf_img]_roilab2cid.json`: mapping from int32 labels in roilab to unique candidate ids. Note: each unique candidate id is represented by one or more (depending on candidate size + local CMF enhancements) unique ROI ids. 

### CMF Label Image Quicklooks 

Path: `/localstore/ang/y[yy]/cmf/ch4/ort/labels/quicklooks`

We generate a pdf quicklook figure to facilitate easy inspection of the generated CMF label images wrt their corresponding RGB+CMF data. Each quicklook contains two subplots: 

* RGB image with CMF overlay w/ enhancements in the (500,1500] ppmm range (top)
* RGB image w/ label image overlay (bottom) 

Circles with radii (50,300)m centered on the (latitude,longitude) provided for each QCed plume/false detection candidate are shown in both subplots to indicate the spatial range of pixels considered in generating each candidate's ROI boundaries. 

### CMF/Label Image Tiles 

Path: `/localstore/ang/y[yy]/cmf/ch4/ort/labels/tiles256`

We train our CNN using CMF image tiles of fixed dimensions (256x256 pixels). Each tile represents either a plume candidate (positive sample), a false enhancement (negative sample), or a background enhancement (background sample). We select these tiles via stratified spatial sampling of the label images. We extract a 256x256 pixel tile centered on each sampled coordinate from the CMF image, and also from its corresponding label image. We save all extracted CMF tiles as 4-channel (R+G+B+CMF) float32 GeoTiffs, and save the coregistered label tiles as png images. 

The following items are generated for each labeled CMF image:

- `tiles256/[cmf_img]/(pos|neg|bg)`: Directories containing positive, negative, and background tiles (GeoTiff) + matching label image tiles (PNG)

- `tiles256/[cmf_img]_tilemap.png`: A map of the same pixel dimensions as the CMF indicating locations of positive, negative and background tiles.
- `tiles256/[cmf_img]_tiles.pdf`: A quicklook showing the RGB+CMF image (top) in context of the locations of positive, negative and background tiles (bottom).
	
<a name="foot1">[1]</a> We use the term "weakly labeled" because the candidate ROI pixel boundaries are not manually drawn by human experts. Rather, experts provide a single (latitude, longitude) location for each candidate. The ROI boundaries are computed based on the intensity of CMF enhancements adjacent to each candidate location.

<a name="foot2">[2]</a> We currently use a minppmm threshold of 250 ppmm for high altitude flightlines (GSD in 8-10 meters), and 500 ppmm for low altitude flightlines with (GSD 3-6 meters).