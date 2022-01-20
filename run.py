from multiprocessing.sharedctypes import Value
import os
import json
import argparse
import requests
from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score, 
    f1_score, 
    roc_auc_score, 
    plot_confusion_matrix, 
    plot_roc_curve, 
)
from sympy import monic

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset

import torchvision
import torchvision.transforms as T
from torchvision.datasets import MNIST, CIFAR10, CIFAR100

import timm
import wandb
from tqdm.auto import tqdm

from pool import ActivePool
from methods import RandomSampling, UncertaintySampling, MarginSampling
from models import MNISTCNN

def load_dataset(name: str):
    train_transform = T.Compose([
        T.ToTensor(), 
        T.RandomHorizontalFlip(0.5), 
        T.RandomRotation((-10, 10))
    ])
    test_transform  = T.Compose([T.ToTensor()])

    if name.lower() == "mnist":
        train_set = MNIST(root="/opt/datasets/mnist", train=True,  transform=T.ToTensor(), download=True)
        test_set  = MNIST(root="/opt/datasets/mnist", train=False, transform=T.ToTensor(), download=True)

    elif name.lower() == "cifar10":
        train_set = CIFAR10(root="/opt/datasets/cifar10", train=True,  transform=train_transform, download=True)
        test_set  = CIFAR10(root="/opt/datasets/cifar10", train=False, transform=test_transform, download=True)
        print(f"lenght of train set {len(train_set)}, test set {len(test_set)}")

    elif name.lower() == "cifar100":
        test_transform  = T.Compose([T.ToTensor()])
        train_set = CIFAR10(root="/opt/datasets/cifar100", train=True,  transform=train_transform, download=True)
        test_set  = CIFAR10(root="/opt/datasets/cifar100", train=False, transform=test_transform, download=True)
        print(f"lenght of train set {len(train_set)}, test set {len(test_set)}")

    else:
        raise ValueError("Not a proper dataset name")

    return train_set, test_set

def get_model(num_classes:int):
    model = timm.create_model("resnet50", pretrained=False, num_classes=num_classes)
    return model

def get_sampler(name: str):
    if name.lower() == "random":
        return RandomSampling
    elif name.lower() == "uncertainty":
        return UncertaintySampling
    elif name.lower() == "margin":
        return MarginSampling

def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu") 

    train_set, test_set = load_dataset(args.dataset)
    
    TRAIN_ACC = []
    TEST_ACC  = []

    num_stages = int(len(train_set) * (1-args.initial_label_rate)) // args.query_size
    max_steps  = (len(train_set) // args.batch_size) * args.max_epochs

    pool  = ActivePool(train_set, batch_size=args.batch_size)
    init_sampler = RandomSampling(None, pool, int(len(train_set)*args.initial_label_rate))
    sampler = get_sampler(args.query_type)(None, pool, args.query_size)

    init_samples = init_sampler()
    pool.update(init_samples)

    run_summary = {"stage": [], "train_acc": [], "test_acc": [], "query_length": [], "query_time": []}

    for stage in range(num_stages):
        
        train_dl = pool.get_labeled_dataloader()
        eval_dl  = pool.get_unlabeled_dataloader()

        model = get_model(num_classes=len(train_set.classes)).to(device)
        sampler.update_model(model)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=0.01)

        epochs = 0
        steps = 0
        training = True

        while training:
            all_targets = []
            all_preds   = []

            model.train()
            for X, y in tqdm(train_dl, desc="train on labeled set"):
                optimizer.zero_grad()

                X = X.to(device)
                y = y.to(device)

                out = model(X)
                loss = criterion(out, y)
                loss.backward()
                optimizer.step()

                pred = torch.argmax(out, dim=1)
                all_preds.extend(pred.tolist())
                all_targets.extend(y.detach().cpu().tolist())

                steps += 1
                if steps > max_steps:
                    training = False
                    break
            
            acc = accuracy_score(all_targets, all_preds)
            TRAIN_ACC.append(acc)

            epochs += 1
            if acc > args.early_stopping_threshold:
                print(f"Early stopped at epoch {epochs+1} with accuracy score {acc:.3f}")
                break

        print(f"Stage {stage} Train accuracy: {acc:.3f}", end=" ")
        run_summary["stage"].append(stage)
        run_summary["train_acc"].append(acc)

        all_targets = []
        all_preds = []

        model.eval()
        with torch.no_grad():
            for X, y in tqdm(eval_dl, desc="evaluation on test set"):
                X = X.to(device)
                y = y.to(device)

                out = model(X)
                loss = criterion(out, y)

                pred = torch.argmax(out, dim=1)
                all_preds.extend(pred.tolist())
                all_targets.extend(y.detach().cpu().tolist())
        
        acc = accuracy_score(all_targets, all_preds)
        TEST_ACC.append(acc)
        print(f"Test accuracy: {acc:.3f}")
        run_summary["test_acc"].append(acc)

        result = sampler()
        pool.update(result)
        print(f"Labeled pool updated with size {len(result.indices)}, time consumed {result.info['time']:.1f}s")
        run_summary["query_length"].append(len(result.indices))
        run_summary["query_time"].append(result.info["time"])

        wandb.log({k: v[-1] for k, v in run_summary.items()})

    with open(f"{args.run_name}_{args.query_type}_{args.dataset}.json") as f:
        json.dump(run_summary, f, ensure_ascii=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--run_name', type=str, default="exp")
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--device', type=str, default="cuda:0")

    parser.add_argument('--max_epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--early_stopping_threshold', type=float, default=0.9)

    parser.add_argument('--query_size', type=int, default=100)
    parser.add_argument('--query_type', type=str, default="random")
    parser.add_argument('--initial_label_rate', type=float, default=0.1)

    parser.add_argument('--wandb_project', type=str, default="active_learning")

    parser.add_argument('-v', '--verbose', action='store_true')
    
    args = parser.parse_args()

    wandb.init(
        project=args.wandb_project,
        name=f"{args.run_name}_{args.query_type}_{args.dataset}",
    )

    main(args)
