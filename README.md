# Methane SourceFinder
Tools & utilities for AVIRIS-NG image processing/product generation for Methane SourceFinder.

## Usage

```
robust_mf.py [-h] [-v] [-k KMODES] [-r] [-f] [--rgb_bands RGB_BANDS] [-m] [-R] [-M MODEL] INPUT LIBRARY OUTPUT

Robust Columnwise Matched Filter with Leave One Out shrinkage estimation via Theiler, 2012.

positional arguments:
  INPUT                 path to input, non-orthocorrected AVIRIS-NG radiance or reflectance image
  LIBRARY               path to target CH4 library file
  OUTPUT                path for output CH4 matched filter image (pixelwise units ppm x meter)

optional arguments:
  -h, --help            show this help message and exit
  -v, --verbose         verbose output
  -k KMODES, --kmodes KMODES
                        number of columnwise modes (k-means clusters)
  -r, --reject          enable multimodal covariance outlier cluster rejection
  -f, --full            regularize multimodal estimates with the full column
                        covariariance
  --rgb_bands RGB_BANDS
                        comma-separated list of RGB channels
  -m, --metadata        save metadata image
  -R, --reflectance     reflectance signature
  -M MODEL, --model MODEL
                        background covariance model (looshrinkage (default)|empirical)
```

## Methane Spectral Library

ang_ch4_unit_3col_425chan.txt: lab measured CH4 transmittance spectrum with columns 
```
[channel index] [wavelength] [CH4 transmittance]
```

## References
- C. Frankenberg, A. K. Thorpe, D. R. Thompson, G. Hulley, E. A. Kort, N. Vance, J. Borchardt, T. Krings, K. Gerilowski, C. Sweeney, S. Conley, B. D. Bue, A. D. Aubrey, S. Hook, and R. O. Green, “Airborne methane remote measurements reveal heavy-tail flux distribution in Four Corners region,” Proceedings of the National Academy of Sciences, 2016.
- D. R. Thompson, I. Leifer, H. Bovensmann, M. Eastwood, M. Fladeland, C. Frankenberg, K. Gerilowski, R. O. Green, S. Kratwurst, T. Krings, B. Luna, and A. K. Thorpe, “Real time remote detection and measurement for airborne imaging spectroscopy: a case study with methane,” Atmospheric Measurement Techniques, 2015.
- J. Theiler, “The incredible shrinking covariance estimator,” SPIE Defense, 2012.
