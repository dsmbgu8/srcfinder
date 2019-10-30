# srcfinder
Tools & utilities for Methane SourceFinder AVIRIS-NG image processing/product generation.

*Usage*
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

*CH4 Spectral Library*
ang_ch4_unit_3col_425chan.txt: lab measured CH4 transmittance spectrum with columns 
```
[channel index] [wavelength] [CH4 transmittance]
```

*References*
J. Theiler, “The incredible shrinking covariance estimator,” SPIE Defense, 2012.
