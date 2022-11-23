import sys
import os
import os.path as op
from pathlib import Path
import argparse

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
from torch import nn
from torch.nn import functional as F
from torchvision import transforms

import rasterio
from archs.googlenet1 import googlenet

class ClampCH4(object):
    """ Preprocessing step for the methane layer """
    def __init__(self, vmin=250, vmax=4000):
        assert isinstance(vmin,int) and isinstance(vmax,int) and vmax > vmin
        self.vmin = vmin
        self.vmax = vmax

    def __call__(self, T):
        return torch.clamp(T, self.vmin, self.vmax)

    def __repr__(self):
        return self.__class__.__name__ + '(vmin={0}, vmax={1})'.format(self.vmin, self.vmax)

class FlightlineShiftStitch(torch.utils.data.Dataset):
    """ Single flightline for shift and stitching
    
    Usage:
    FlightlineShiftStitch(flightlinepath, transform, scale)
    """
    
    def __init__(self, flightline, transform, scale=32):
        self.flightline = flightline
        self.x = np.expand_dims(rasterio.open(self.flightline).read(4), axis=0)
        self.transform = transform
        self.scale = scale
    
    def __len__(self):
        return self.scale ** 2
    
    def __getitem__(self, idx):        
        # Calculate padding for this index
        top = idx // self.scale
        left = idx % self.scale
        
        t = torch.as_tensor(self.x, dtype=torch.float)
        if self.transform is not None:
            t = self.transform(t)
        
        t = transforms.Pad([self.scale-1-left, self.scale-1-top, left, top], fill=0, padding_mode='constant')(t)
        
        return t 

def stitch_stack(fl_shape, predstack, scale=32):
    stitched = np.zeros(shape=fl_shape)
    
    for i in range(scale**2):
        top = i // scale
        left = i % scale
        
        inshape = stitched[top::scale, left::scale].shape
        stitched[top::scale, left::scale] = predstack[i, :inshape[0], :inshape[1]]
    
    return stitched


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a flightline saliency map with a FCN."
    )

    parser.add_argument('flightline',       help="Filepaths to flightline ENVI IMG.",
                                            type=str)
    parser.add_argument('--model', '-m',    help="Model to use for prediction.",
                                            default="COVID_QC",
                                            choices=["COVID_QC", "CalCH4_v8", "Permian_QC", "CalCh4_v8+COVID_QC+Permian_QC"])
    parser.add_argument('--gpus', '-g',     help="GPU devices for inference. -1 for CPU.",
                                            nargs='+',
                                            default=[-1],
                                            type=int)
    parser.add_argument('--batch', '-b',    help="Batch size per device.",
                                            default=32,
                                            type=int)
    parser.add_argument('--output', '-o',   help="Output directory for generated saliency maps.",
                                            default=".",
                                            type=str)

    args = parser.parse_args()


    # Initial model setup/loading
    print("[STEP] MODEL INITIALIZATION")

    print("[INFO] Finding model weightpath.")
    weightpath = op.join(Path(__file__).parent.resolve(), 'models', f"{args.model}.pt")
    if op.isfile(weightpath):
        print(f"[INFO] Found {weightpath}.")
    else:
        print(f"[INFO] Model not found at {weightpath}, exiting.")
        sys.exit(1)

    print("[INFO] Initializing pytorch device.")
    if args.gpus == [-1]:
        # CPU
        device = torch.device('cpu')
    else:
        if not torch.cuda.is_available():
            print("[ERR] CUDA not found, exiting.")
            sys.exit(1)
        
        # Set first device
        device = torch.device(f"cuda:{args.gpus[0]}")

    print("[INFO] Loading model.")
    model = googlenet(pretrained=False, num_classes=2, init_weights=False)
    model.load_state_dict(torch.load(weightpath))
    
    # FCN model setup
    print("[INFO] Converting CNN to FCN.")
    fcn = nn.Sequential(*list(model.children())[:-5]).to(device)
    fcn.add_module('final_conv', nn.Conv2d(1024, 2, kernel_size=1).to(device))
    fcn.final_conv.weight.data.copy_(model.fc.weight.data[:,:,None,None])
    fcn.final_conv.bias.data.copy_(model.fc.bias.data)

    if len(args.gpus) > 1:
        # Multi-GPU
        fcn = fcn.to(device)
        fcn = nn.DataParallel(fcn, device_ids=args.gpus)
    else:
        # Single-GPU or CPU
        fcn = fcn.to(device)
    
    fcn.eval()

    print("[INFO] Initializing Dataloader.")
    # Transform and dataloader
    if args.model == "COVID_QC":
        transform = transforms.Compose([
            ClampCH4(vmin=0, vmax=4000),
            transforms.Normalize(
                mean=[110.6390],
                std=[183.9152]
            )]
        )
    elif args.model == "CalCH4_v8":
        transform = transforms.Compose([
            ClampCH4(vmin=0, vmax=4000),
            transforms.Normalize(
                mean=[140.6399],
                std=[237.5434]
            )]
        )
    elif args.model == "Permian_QC":
        transform = transforms.Compose([
            ClampCH4(vmin=0, vmax=4000),
            transforms.Normalize(
                mean=[100.2635],
                std=[158.7060]
            )]
        )
    elif args.model == "CalCh4_v8+COVID_QC+Permian_QC":
        transform = transforms.Compose([
            ClampCH4(vmin=0, vmax=4000),
            transforms.Normalize(
                mean=[115.0],
                std=[190.0]
            )]
        )

    dataloader = torch.utils.data.DataLoader(
        FlightlineShiftStitch(
            args.flightline,
            transform=transform,
            scale=256
        ),
        batch_size=args.batch * len(args.gpus),
        shuffle=False,
        num_workers=8
    )

    print("[STEP] MODEL PREDICTION")

    # Run shift predictions
    allpred = []
    for batch in tqdm(dataloader, desc="FCN Pred"):
        inputs = batch.to(device)
        with torch.no_grad():
            preds = fcn(inputs)
            preds = torch.nn.functional.softmax(preds, dim=1)
            allpred += [x[1] for x in preds.cpu().detach().numpy()]
    allpred = np.array(allpred)

    # Stitch
    print("[INFO] Stitching shifts.")
    dataset = rasterio.open(args.flightline)
    array = dataset.read(4)
    allpred = stitch_stack(array.shape, allpred, scale=256)
    allpred[array == -9999] = -9999

    # Save
    print("[STEP] RESULT EXPORT")
    with rasterio.Env():
        profile = dataset.profile

        profile.update(
            dtype=rasterio.float32,
            count=1,
            compress='lzw'
        )

        print(f"[INFO] Saving to", op.join(args.output, f"{Path(args.flightline).stem}_saliency.img"))
        with rasterio.open(op.join(args.output, f"{Path(args.flightline).stem}_saliency.img"), 'w', **profile) as dst:
            dst.write(allpred.astype(rasterio.float32), 1)

    print("Done!")
