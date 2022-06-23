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
from archs.googlenet import googlenet

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
        self.x = np.expand_dims(rasterio.open(self.flightline).read(4).T, axis=0)
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
    parser = argparse.ArgumentParser(description="Convert a CNN to an FCN for flightline predictions.")
    parser.add_argument('flightline',       help="Filepath to flightline img")
    args = parser.parse_args()

    # Initial model setup/loading
    print("Loading CNN model...")
    device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
    model = googlenet(pretrained=False, num_classes=2, init_weights=True).to(device)
    weights = "/scratch/jakelee/ch4/results/20220110_232225_814587_COVID_QC_augB/weights/29_20220110_232225_814587_COVID_QC_augB_weights.pt"
    model.load_state_dict(torch.load(weights))
    model.eval()

    # FCN model setup
    print("Converting CNN to FCN...")
    fcn = nn.Sequential(*list(model.children())[:-5]).to(device)
    fcn.add_module('final_conv', nn.Conv2d(1024, 2, kernel_size=1).to(device))
    fcn.final_conv.weight.data.copy_(model.fc.weight.data[:,:,None,None])
    fcn.final_conv.bias.data.copy_(model.fc.bias.data)

    # Transform and dataloader
    print("Setting up Dataloader...")
    transform = transforms.Compose([
        ClampCH4(vmin=250, vmax=4000),
        transforms.Normalize(
            mean=[289.2123],
            std=[109.5958]
        )]
    )

    dataloader = torch.utils.data.DataLoader(
        FlightlineShiftStitch(
            args.flightline,
            transform=transform,
            scale=32
        ),
        batch_size=8,
        shuffle=False,
        num_workers=4
    )

    # Run shift predictions
    allpred = []
    for batch in tqdm(dataloader, desc="Predicting shifts"):
        inputs = batch.to(device)
        with torch.no_grad():
            preds = fcn(inputs)
            preds = torch.nn.functional.softmax(preds, dim=1)
            allpred += [x[1] for x in preds.cpu().detach().numpy()]
    allpred = np.array(allpred)

    # Export for debug
    #np.save("predstack.npy", allpred)

    # Stitch
    print("Stitching shifts...")
    display_x = rasterio.open(args.flightline).read(4).T
    display_x = np.clip(display_x, 0, 4000)
    stitched = stitch_stack(display_x.shape, allpred, scale=32)

    # Save
    fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(12,8))
    ax1.imshow(display_x, vmin=0, vmax=4000)
    ax2.imshow(stitched, vmin=0, vmax=1.0)
    fig.savefig(f"{Path(args.flightline).stem}-pred.png")

    print("Done!")
