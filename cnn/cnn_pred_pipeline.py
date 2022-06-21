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

class FlightlineConvolve(torch.utils.data.Dataset):
    """ Single flightline for exhaustive CNN convolution"""

    def __init__(self, flightline, transform, dim=256):
        self.flightline = flightline
        self.transform = transform

        x = np.expand_dims(rasterio.open(self.flightline).read(4).T, axis=0)
        self.inshape = x.shape
        print(self.inshape)

        x = torch.as_tensor(x, dtype=torch.float)
        if self.transform is not None:
            x = self.transform(x)
        self.x = transforms.Pad([dim//2, dim//2, (dim//2)-1, (dim//2)-1], fill=0, padding_mode='constant')(x)

        self.dim = dim

    def __len__(self):
        return int(self.inshape[1] * self.inshape[2])
    
    def __getitem__(self, idx):
        row = idx // self.inshape[2]
        col = idx % self.inshape[2]

        return self.x[:,row:row+self.dim, col:col+self.dim]

        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description = "Generate a flightline saliency map with a CNN."
    )

    parser.add_argument('flightline',       help="Filepaths to flightline ENVI IMG.",
                                            type=str)
    parser.add_argument('--model', '-m',    help="Model to use for prediction.",
                                            default="COVID_QC",
                                            choices=["COVID_QC", "CalCH4_v8", "Permian_QC"])
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
    
    if len(args.gpus) > 1:
        # Multi-GPU
        model = nn.DataParallel(model, device_ids=args.gpus)
    else:
        # Single-GPU or CPU
        model = model.to(device)
    
    model.eval()

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

    dataloader = torch.utils.data.DataLoader(
        FlightlineConvolve(
            args.flightline,
            transform=transform,
        ),
        batch_size=args.batch * len(args.gpus),
        shuffle=False,
        num_workers=8
    )

    # Run shift predictions
    allpred = []
    for batch in tqdm(dataloader, desc="Predicting shifts"):
        inputs = batch.to(device)
        with torch.no_grad():
            preds = model(inputs)
            preds = torch.nn.functional.softmax(preds, dim=1)
            allpred += [x[1] for x in preds.cpu().detach().numpy()]

    # Save
    dataset = rasterio.open(args.flightline)
    array = dataset.read(4)

    allpred = np.array(allpred).reshape(array.shape)
    allpred[array == -9999] = -9999

    with rasterio.Env():
        profile = dataset.profile

        profile.update(
            dtype=rasterio.float32,
            count=1,
            compress='lzw'
        )

        with rasterio.open(op.join(args.output, f"{Path(args.flightline).stem}_saliency.img"), 'w', **profile) as dst:
            dst.write(allpred.astype(rasterio.float32), 1)

    print("Done!")
