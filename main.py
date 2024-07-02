import torch
import torchvision
from matplotlib import pyplot as plt
from tsimcne.tsimcne import TSimCNE

import wandb
import argparse

def main():
    
    # argparse
    parser = argparse.ArgumentParser(description='t-SimCNE')
    parser.add_argument('--epochs', type=int, nargs='+', default=[1000, 50, 450], help='epochs for each stage')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--devices', type=int, default=1, help='number of devices')
    
    args = parser.parse_args()
    
    
    
    wandb.init(project="tsimcne")

    # get the cifar dataset (make sure to adapt `data_root` to point to your folder)
    data_root = "/zangzelin/data/cifar-10"
    dataset_train = torchvision.datasets.CIFAR10(
        root=data_root,
        download=True,
        train=True,
    )
    dataset_test = torchvision.datasets.CIFAR10(
        root=data_root,
        download=True,
        train=False,
    )
    dataset_full = torch.utils.data.ConcatDataset([dataset_train, dataset_test])

    # create the object (here we run t-SimCNE with fewer epochs
    # than in the paper; there we used [1000, 50, 450]).
    # tsimcne = TSimCNE(total_epochs=[500, 50, 250])
    tsimcne = TSimCNE(
        total_epochs=args.epochs,
        devices=args.devices,
        )

    # train on the augmented/contrastive dataloader (this takes the most time)
    tsimcne.fit(dataset_full)

    # map the original images to 2D
    Y = tsimcne.transform(dataset_full)

    # get the original labels from the dataset
    labels = [lbl for img, lbl in dataset_full]

    # plot the data
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(*Y.T, c=labels, cmap="tab10", s=1, )
    wandb.log({"tsne": plt})
    wandb.finish()