# Carbon Mapper CNN

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

## FCN Pipeline

Not ready yet, do not use.
