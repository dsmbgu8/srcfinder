# CNN Model Training

## Description

This is a simplified version of the trainings script used to train the plume
detection models.

## Dependencies

* `tqdm`
* `sklearn`, `pytorch`, `torchvision`
* `numpy`, `matplotlib`, `PIL`
* `rasterio`, `GDAL`

## References

* [GoogLeNet](https://arxiv.org/pdf/1409.4842.pdf)
* [Sharpness Aware Minimization (SAM)](https://arxiv.org/pdf/2010.01412.pdf)

## Expected Dataset Structure

```
.
├── ang20200708t192518_ch4mf_v2y1_img
│   ├── bg    // background tiles (0)
│   ├── neg   // false positive tiles (-1)
│   └── pos   // true positive tiles (1)
│       ├── *.tif           // these are ENVI IMG pretending to be GTiff
│       ├── *.tif.aux.xml
│       └── *.hdr
├── train.csv         // tilepath relative to ./ , label [-1,0,1] for training set
├── test.csv          // same for test set
└── data_labels.csv   // same for entire dataset
```

## Usage

```
$ python experiment_script_all.py -h
usage: experiment_script_all.py [-h] [--lr LR] [--augment AUGMENT]
                                [--crop CROP] [--epochs EPOCHS]
                                [--outroot OUTROOT] [--no-sam] [--gpu GPU]
                                [--train-all]
                                dataroot {CalCH4_v8,COVID_QC,Permian_QC}

Train a classification model on tiled methane data.

positional arguments:
  dataroot              Directory path to dataset root
  {CalCH4_v8,COVID_QC,Permian_QC}
                        Campaign to train & test on

optional arguments:
  -h, --help            show this help message and exit
  --lr LR               Learning rate
  --augment AUGMENT     Data augmentation option
  --crop CROP           Center-crop the input tiles
  --epochs EPOCHS       Epochs for training
  --outroot OUTROOT     Root of output directories
  --no-sam              Disable SAM
  --gpu GPU             Specify GPU index to use
  --train-all           Train on the entire dataset
```