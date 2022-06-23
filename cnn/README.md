# Carbon Mapper CNN

This directory contains pipelines for deep learning detection of methane plumes.

## CNN Pipeline

```
$ python cnn_pred_pipeline.py -h
usage: cnn_pred_pipeline.py [-h] [--model {COVID_QC,CalCH4_v8,Permian_QC}]
                            [--gpus GPUS [GPUS ...]] [--batch BATCH]
                            [--output OUTPUT]
                            flightline

Generate a flightline saliency map with a CNN.

positional arguments:
  flightline            Filepaths to flightline ENVI IMG.

optional arguments:
  -h, --help            show this help message and exit
  --model {COVID_QC,CalCH4_v8,Permian_QC}, -m {COVID_QC,CalCH4_v8,Permian_QC}
                        Model to use for prediction.
  --gpus GPUS [GPUS ...], -g GPUS [GPUS ...]
                        GPU devices for inference. -1 for CPU.
  --batch BATCH, -b BATCH
                        Batch size per device.
  --output OUTPUT, -o OUTPUT
                        Output directory for generated saliency maps.
```

### Summary

This pipeline generates a saliency map for each flightline by predicting a value
for each pixel with a CNN.

The saliency map is saved as an ENVI IMG in the directory specified by
`--output`. Its default value is `./`, the current working directory. The
saliency map has float values ranging from `[0,1]`, with `1` indicating the
presence of a methane plume.

### Dependencies

The following packages must be installed via `pip`, along with the appropriate
GPU CUDA drivers.

* `numpy`
* `matplotlib`
* `tqdm`
* `rasterio`
* `pytorch`
* `torchvision`

### Examples

These examples were run on a 32-core `Intel(R) Xeon(R) CPU E5-2667 v4 @ 3.20GHz`
server with `126 GB` of RAM and two Tesla M60 cards, effectively 4 GPUs with 
`2048` CUDA cores and `8GB` of VRAM each.

These examples include benchmarks for expected performance with different
hardware configurations. Notice that more GPUs and larger batch sizes do not
always result in the fastest runtime - there is some tradeoff due to memory
management.

#### CPU Inference

```bash
// CPU inference with batch size 8
$ python cnn_pred_pipeline.py samples/ang20200924t211102_ch4mf_v2y1_img -m COVID_QC -g -1 -b 8 
[STEP] MODEL INITIALIZATION
[INFO] Finding model weightpath.
[INFO] Found /home/jakelee/2022/srcfinder/cnn/models/COVID_QC.pt.
[INFO] Initializing pytorch device.
[INFO] Loading model.
[INFO] Initializing Dataloader.
(1, 669, 2801)
[STEP] MODEL PREDICTION
Predicting shifts...
// ETA ~8 hours
```

```bash
// CPU inference with batch size 32
$ python cnn_pred_pipeline.py samples/ang20200924t211102_ch4mf_v2y1_img -m COVID_QC -g -1 -b 32
// ETA ~7 hours
```

#### Single GPU Inference

```bash
// Single GPU inference with batch size 8
$ python cnn_pred_pipeline.py samples/ang20200924t211102_ch4mf_v2y1_img -m COVID_QC -g 0 -b 8
// ETA ~2.3 hours
```

```bash
// Single GPU inference with batch size 32
$ python cnn_pred_pipeline.py samples/ang20200924t211102_ch4mf_v2y1_img -m COVID_QC -g 0 -b 32
// ETA ~1.3 hours
```

```bash
// Single GPU inference with batch size 512
$ python cnn_pred_pipeline.py samples/ang20200924t211102_ch4mf_v2y1_img -m COVID_QC -g 0 -b 512
// ETA ~1.3 hours
```

#### Quad GPU Inference

```bash
// Quad-GPU inference with batch size 8 each
$ python cnn_pred_pipeline.py samples/ang20200924t211102_ch4mf_v2y1_img -m COVID_QC -g 0 1 2 3 -b 8
// ETA ~7.5 hours
```

```bash
// Quad-GPU inference with batch size 32 each
$ python cnn_pred_pipeline.py samples/ang20200924t211102_ch4mf_v2y1_img -m COVID_QC -g 0 1 2 3 -b 32
// ETA ~2 hours
```

```bash
// Quad-GPU inference with batch size 512 each
$ python cnn_pred_pipeline.py samples/ang20200924t211102_ch4mf_v2y1_img -m COVID_QC -g 0 1 2 3 -b 512
// ETA ~0.5 hours
```

## FCN Pipeline

Not ready yet, do not use.
