""" Train/Test Experiment Script

v1: 2021-11-07 jakelee
v2: 2021-11-14 jakelee
v3: 2022-01-10 jakelee
v4: 2022-04-10 jakelee

Jake Lee, jakelee, jake.h.lee@jpl.nasa.gov
"""
import sys
import os
import os.path as op
from pathlib import Path
import csv
import argparse
from datetime import datetime
from tqdm import tqdm
import time

import torch
import torch.nn     as nn
from torch.nn       import CrossEntropyLoss
from torchvision    import transforms
from torch.optim    import SGD

sys.path.append('../')
from archs.googlenet1 import googlenet

from sklearn.metrics import precision_recall_curve, classification_report
import matplotlib.pyplot as plt
import numpy as np
import PIL
import rasterio

sys.path.append('./sam')
from sam import SAM
from example.utility.bypass_bn import disable_running_stats, enable_running_stats
from example.utility.step_lr import StepLR


class ClampMethaneTile(object):
    """ Preprocessing step for methane match filter tiles

    Parameters
    ----------
    ch4min: int
        Minimum clip for the methane layer. Defaults to 250.
        Applied to channel 1 if single-channel, 4 if quad-channel.
    ch4max: int
        Maximum clip for the methane layer. Defaults to 4000.
        Applied to channel 1 if single-channel, 4 if quad-channel.
    rgbmin: int
        Minimum clip for the RGB layers. Defaults to 0.
        Not applied if single-channel, channels 1, 2, 3 if quad-channel.
    rgbmax: int
        Maximum clip for the RGB layers. Defaults to 20.
        Not applied if single-channel, channels 1, 2, 3 if quad-channel.
    
    Returns
    -------
    Preprocessed/clipped tensor array
    """
    def __init__(self, ch4min=250, ch4max=4000, rgbmin=0, rgbmax=20):
        assert isinstance(ch4min,int) and isinstance(ch4max,int) and ch4max > ch4min
        assert isinstance(rgbmin,int) and isinstance(ch4max,int) and ch4max > ch4min

        self.ch4min = ch4min
        self.ch4max = ch4max
        self.rgbmin = rgbmin
        self.rgbmax = rgbmax
    
    def __call__(self, img):
        # We only expect 1 or 4 channels (CH4 or RGB+CH4)
        assert img.shape[0] == 1 or img.shape[0] == 4

        if img.shape[0] == 1:
            return torch.clamp(img, self.ch4min, self.ch4max)
        elif img.shape[0] == 4:
            img[:3] = torch.clamp(img[:3], self.rgbmin, self.rgbmax)
            img[3] = torch.clamp(img[3], self.ch4min, self.ch4max)
            return img
        
    def __repr__(self):
        return self.__class__.__name__ + f'(ch4min={self.ch4min}, ch4max={self.ch4max}, rgbmin={self.rgbmin}, rgbmax={self.rgbmax})'

class TiledDatasetClass1Ch(torch.utils.data.Dataset):
    """ Classification dataset only using the methane channel
    
    Usage:
    TiledDatasetSeg1Ch([[path, label], [path, label], ...], ...)
    """

    def __init__(self, dataroot, datacsv, transform):
        self.dataroot = dataroot
        self.datacsv = datacsv
        self.transform = transform

    def __len__(self):
        return len(self.datacsv)

    def __getitem__(self, idx):
        # Absolute path to image
        x_path = self.datacsv[idx][0]
        # Correct absolute path to relative path
        x_path = op.join(self.dataroot, *Path(x_path).parts[-3:])

        # 1/0 label (-1 is 0)
        y = 1 if int(self.datacsv[idx][1]) == 1 else 0

        x = rasterio.open(x_path).read(4)
        x = np.expand_dims(x, axis=0)

        x = torch.as_tensor(x, dtype=torch.float)
        if self.transform is not None:
            x = self.transform(x)

        return (x, y)

def get_augment(mean, std, augment="default", crop=256):
    """Define dataset augmentation for high-resolution data"""

    if augment == "augA" or augment == "default":
        # no aug
        transform = transforms.Compose([
            ClampMethaneTile(ch4min=0, ch4max=4000, rgbmin=0, rgbmax=20),
            transforms.CenterCrop(crop),
            transforms.Normalize(
                mean=mean,
                std=std
            )
        ])
    elif augment == "augB":
        # flip aug
        transform = transforms.Compose([
            ClampMethaneTile(ch4min=0, ch4max=4000, rgbmin=0, rgbmax=20),
            transforms.CenterCrop(crop),
            transforms.Normalize(
                mean=mean,
                std=std
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5)
        ])
    elif augment == "augC":
        # affine aug
        transform = transforms.Compose([
            ClampMethaneTile(ch4min=0, ch4max=4000, rgbmin=0, rgbmax=20),
            transforms.CenterCrop(crop),
            transforms.Normalize(
                mean=mean,
                std=std
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomAffine(
                degrees=(-180,180),
                translate=(0.1, 0.1),
                resample=PIL.Image.BILINEAR)
        ])

    return transform


def build_dataloader(dataroot, campaign=None, augment="default", mode="train", shuffle=True, crop=256):
    """ Build a pytorch dataloader based on provided arguments
    
    These preprocessing steps include calculated mean and standard deviation
    values for the _training set_ of each tiled dataset.

    dataroot: str
        Path to parent directory of datasets
    campaign: str
        Name of campaign; subdir of dataroot
    augment: str
        Dataset augmentation to use. Defaults to "default".
    mode: [train, test, all]
        Whether to load the train or test dataset. Defaults to train.
    shuffle: bool
        Whether to shuffle the dataset. Defaults to True.
    crop: int
        Center cropping dimensions for training the model for FCN conversion. Defaults to 256.
    """


    # Load data CSV
    datarows = []
    if mode == "train":
        train_csv = op.join(dataroot, campaign, "train.csv")
    elif mode == "test":
        train_csv = op.join(dataroot, campaign, "test.csv")
    elif mode == "all":
        train_csv = op.join(dataroot, campaign, "data_labels.csv")

    with open(train_csv, 'r') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            datarows.append(row)
    
    # Get loss weights
    all_labels = [1 if int(r[1]) == 1 else 0 for r in datarows]
    loss_weights = [1, (len(all_labels) - sum(all_labels)) / sum(all_labels)]

    # Define transforms and dataset class
    if campaign == "CalCH4_v8":
        # NOTE: (0, 4000) on train
        transform = get_augment([140.6399], [237.5434], augment, crop)
        dataset = TiledDatasetClass1Ch(op.join(dataroot, campaign), datarows, transform)
    elif campaign == "COVID_QC":
        # NOTE: (0, 4000) on train
        transform = get_augment([110.6390], [183.9152], augment, crop)
        dataset = TiledDatasetClass1Ch(op.join(dataroot, campaign), datarows, transform)
    elif campaign == "Permian_QC":
        # NOTE: (0, 4000) on train
        transform = get_augment([100.2635], [158.7060], augment, crop)
        dataset = TiledDatasetClass1Ch(op.join(dataroot, campaign), datarows, transform)
    else:
        raise Exception(f"Undefined 1ch campaign: {campaign}")

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=16,
        shuffle=shuffle,
        num_workers=16
    )

    return dataloader, loss_weights


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a classification model on tiled methane data.")

    parser.add_argument('dataroot',         help="Directory path to dataset root")
    parser.add_argument('campaign',         choices=["COVID_v8", "CalCH4_v8", "COVID_v8_30m", "CalCH4_v8_30m", "COVID_QC", "COVID_QC_30m", "Permian_QC"],
                                            help="Campaign to train & test on")
    parser.add_argument('--lr',             type=float,
                                            help="Learning rate",
                                            default=0.0001)
    parser.add_argument('--augment',        help="Data augmentation option",
                                            default="default")
    parser.add_argument('--crop',           type=int,
                                            help="Center-crop the input tiles",
                                            default=256)
    parser.add_argument('--epochs',         type=int,
                                            default=100,
                                            help="Epochs for training")
    parser.add_argument('--outroot',        default="cnn_output",
                                            help="Root of output directories")
    parser.add_argument('--no-sam',         action='store_true',
                                            help="Disable SAM")
    parser.add_argument('--gpu',            type=int,
                                            default=0,
                                            help="Specify GPU index to use")
    parser.add_argument('--train-all',      action='store_true',
                                            help="Train on the entire dataset")

    args = parser.parse_args()


    ## Set up output directories and files
    expname = f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_{args.campaign}_{args.augment}_{'4ch' if args.use_rgb else '1ch'}_{'all' if args.train_all else 'train'}_{args.crop}"

    outdir = op.join(args.outroot, expname)
    if not op.isdir(outdir):
        os.mkdir(outdir)

    outweightdir = op.join(outdir, "weights")
    if not op.isdir(outweightdir):
        os.mkdir(outweightdir)
    
    batch_losses = [["epoch", "batch", "loss"]]
    train_epoch_losses = [["epoch", "mean train loss"]]
    val_epoch_losses = [["epoch", "mean val loss"]]

    outbatchcsv = op.join(outdir, "batch_losses.csv")
    outepochcsv = op.join(outdir, "epoch_losses.csv")
    outvalcsv = op.join(outdir, "val_losses.csv")

    ## Device
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    ## Load Data & Get Loss Weights
    train_loader, loss_weights = build_dataloader(args.dataroot,
                                                campaign=args.campaign,
                                                augment=args.augment,
                                                mode="all" if args.train_all else "train",
                                                shuffle=True,
                                                crop=args.crop)

    val_loader, _ = build_dataloader(args.dataroot,
                                    campaign=args.campaign,
                                    augment=args.augment,
                                    mode="test",
                                    shuffle=False,
                                    crop=args.crop)

    ## Load Model
    model = googlenet(pretrained=False, num_classes=2, init_weights=True).to(device)

    ## CONSTANTS
    lr = args.lr

    ## Set up optimizer
    if not args.no_sam:
        print("Using SAM")
        optimizer = SAM(model.parameters(),
                        SGD,
                        rho=2.0,
                        adaptive=True,
                        lr=lr,                  # kwargs for SGD
                        momentum=0.9,
                        weight_decay=0.0005)
    else:
        optimizer = SGD(model.parameters(),
                lr=lr, momentum=0.9, weight_decay=0.0005)

    ## Scheduler
    scheduler = StepLR(optimizer, lr, args.epochs)

    ## Loss Function
    print(f"Using class weights {loss_weights}")
    ce_loss = CrossEntropyLoss(weight=torch.as_tensor(loss_weights, device=device))

    ## Train
    for epoch in range(args.epochs):
        model.train()

        start = time.time()
        epoch_loss = 0
        for iter, batch in enumerate(train_loader):
            inputs, targets = (b.to(device) for b in batch)

            if not args.no_sam:
                # first forward-backward step
                enable_running_stats(model)
                predictions = model(inputs).logits
                # predictions = model(inputs)
                loss = ce_loss(predictions, targets)

                loss.mean().backward()
                optimizer.first_step(zero_grad=True)

                # second forward-backward step
                disable_running_stats(model)
                ce_loss(model(inputs).logits, targets).mean().backward()
                # ce_loss(model(inputs), targets).mean().backward()
                optimizer.second_step(zero_grad=True)
            else:
                optimizer.zero_grad()
                predictions = model(inputs).logits
                # predictions = model(inputs)
                loss = ce_loss(predictions, targets)
                loss.backward()
                optimizer.step()

            with torch.no_grad():
                epoch_loss += loss
                print(f"epoch {epoch}, batch {iter}/{len(train_loader)}, train loss {loss.cpu()}")
                batch_losses.append([epoch, iter, loss.cpu().item()])

                scheduler(epoch)

        end = time.time()
        print(f"Epoch {epoch} took {end-start} seconds")

        epoch_loss = epoch_loss.cpu() / len(train_loader)
        print(f"Epoch {epoch} loss {epoch_loss}")
        train_epoch_losses.append([epoch, epoch_loss.item()])

        # Validate
        model.eval()
        val_epoch_loss = 0
        for iter, batch in enumerate(val_loader):
            inputs, targets = (b.to(device) for b in batch)

            with torch.no_grad():
                predictions = model(inputs)
                loss = ce_loss(predictions, targets)
                val_epoch_loss += loss
                print(f"epoch {epoch}, batch {iter}/{len(val_loader)}, val loss {loss.cpu()}")
        
        val_epoch_loss = val_epoch_loss.cpu() / len(val_loader)
        print(f"Epoch {epoch} val loss {val_epoch_loss}")
        val_epoch_losses.append([epoch, val_epoch_loss.item()])

        # Save Weights
        if (epoch + 1) % 5 == 0:
            weightpath = op.join(outweightdir, f"{epoch}_{expname}_weights.pt")
            torch.save(model.state_dict(), weightpath)

        fig, ax = plt.subplots()
        ax.plot(range(epoch+1), np.array(train_epoch_losses[1:])[:,1], label='Train')
        ax.plot(range(epoch+1), np.array(val_epoch_losses[1:])[:,1], label='Val')
        #ax.set_yscale('log')
        ax.grid()

        ax.legend()
        ax.set_xlabel("Epochs")
        ax.set_ylabel("CE Loss")
        ax.set_title(f"{expname} Loss Curve")
        fig.savefig(op.join(outdir, 'loss_curve.png'), dpi=300)

    with open(outbatchcsv, 'w') as f:
        writer = csv.writer(f)
        writer.writerows(batch_losses)
    
    with open(outepochcsv, 'w') as f:
        writer = csv.writer(f)
        writer.writerows(train_epoch_losses)
    
    with open(outvalcsv, 'w') as f:
        writer = csv.writer(f)
        writer.writerows(val_epoch_losses)

    ## Test
    model.eval()

    ## Load Data
    dataloader, _ = build_dataloader(args.dataroot,
                                    campaign=args.campaign,
                                    augment=args.augment,
                                    mode="all" if args.train_all else "train",
                                    shuffle=False,
                                    crop=args.crop)

    # Pred training data set
    train_true = []
    train_pred = []
    for iter, batch in tqdm(enumerate(dataloader), desc="train set pred", total=len(dataloader)):
        inputs, targets = (b.to(device) for b in batch)

        with torch.no_grad():
            predictions = model(inputs)
            probabilities = torch.nn.functional.softmax(predictions, dim=1)
            train_true += targets.cpu().tolist()
            train_pred += [x[1] for x in probabilities.cpu().tolist()]

    with open(op.join(outdir, 'train_predictions.csv'), 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['pred', 'label'])
        for pred, label in zip(train_pred, train_true):
            writer.writerow([pred, label])
    
    train_prec, train_recall, train_thresh = precision_recall_curve(train_true, train_pred)
    train_f1 = [2 * (p * r) / (p + r) for p, r in zip(train_prec, train_recall)]
    best_thresh = train_thresh[np.argmax(train_f1)]

    # Classification Report
    print(classification_report(train_true, [1 if x > best_thresh else 0 for x in train_pred]))
    with open(op.join(outdir, 'train_report.txt'), 'w') as f:
        f.write(classification_report(train_true, [1 if x > best_thresh else 0 for x in train_pred]))

    # Plot
    fig, ax = plt.subplots()
    ax.plot(train_recall, train_prec)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_ylim(0, 1)
    ax.set_xlim(0, 1)
    ax.set_title(f"{expname} Epoch {args.epochs}")
    fig.savefig(op.join(outdir, 'train_PRcurve.png'), dpi=300)

    # Pred training data set
    dataloader, _ = build_dataloader(args.dataroot,
                                    campaign=args.campaign,
                                    augment=args.augment,
                                    mode="test",
                                    shuffle=False,
                                    crop=args.crop)

    val_true = []
    val_pred = []
    for iter, batch in tqdm(enumerate(dataloader), desc="val set pred", total=len(dataloader)):
        inputs, targets = (b.to(device) for b in batch)

        with torch.no_grad():
            predictions = model(inputs)
            probabilities = torch.nn.functional.softmax(predictions, dim=1)
            val_true += targets.cpu().tolist()
            val_pred += [x[1] for x in probabilities.cpu().tolist()]
    
    with open(op.join(outdir, 'val_predictions.csv'), 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['pred', 'label'])
        for pred, label in zip(val_pred, val_true):
            writer.writerow([pred, label])
    
    # Classification Report
    print(classification_report(val_true, [1 if x > best_thresh else 0 for x in val_pred]))
    with open(op.join(outdir, 'val_report.txt'), 'w') as f:
        f.write(classification_report(val_true, [1 if x > best_thresh else 0 for x in val_pred]))
