from typing import List, Dict, Any
from collections import defaultdict
import os
import sys
import json
import logging
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
from methods import BALD, ActiveQuery, EntropySampling, RandomSampling, UncertaintySampling, MarginSampling
from models import MNISTCNN
from utils import QueryResult, set_all_seeds

# Global logger settings
logger = logging.getLogger()
logger.setLevel(logging.INFO)
stream_handler = logging.StreamHandler(sys.stdout)
file_handler = logging.FileHandler(filename="log.txt")

formatter = logging.Formatter(fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

stream_handler.setLevel(logging.INFO)
file_handler.setLevel(logging.DEBUG)

stream_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

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
        test_set  = CIFAR10(root="/opt/datasets/cifar10", train=False, transform=test_transform,  download=True)
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
    model = timm.create_model("resnet18", pretrained=False, num_classes=num_classes)
    return model

def get_sampler(name: str):
    if name.lower() == "random":
        return RandomSampling
    elif name.lower() == "uncertainty":
        return UncertaintySampling
    elif name.lower() == "margin":
        return MarginSampling
    elif name.lower() == "entropy":
        return EntropySampling
    elif name.lower() == "bald":
        return BALD

def log_metrics(prefix: str, metrics: Dict[str, Any]) -> Dict[str, Any]:
    for k in list(metrics.keys()):
        if not k.startswith(f"{prefix}/"):
            metrics[f"{prefix}/{k}"] = metrics.pop(k)
    # wandb
    wandb.log(metrics)
    return metrics

def calc_entropy(p: np.ndarray):
    v = p * np.log(p)
    v[p == 0.0] = 0.0
    return -np.sum(v)

def main(args):

    if args.seed is not None:
        set_all_seeds(args.seed)

    if not os.path.isdir(args.logging_path):
        os.makedirs(args.logging_path)
        logger.warning(f"Logging path {os.path.abspath(args.logging_path)} has been created!")
    
    if not os.path.isdir(args.save_path):
        os.makedirs(args.save_path)
        logger.warning(f"Model save path {os.path.abspath(args.save_path)} has been created!")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu") 

    train_set, test_set = load_dataset(args.dataset)
    test_dl = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    num_classes = len(train_set.classes)
    num_stages = int(len(train_set) * (1-args.initial_label_rate)) // args.query_size
    max_steps  = (len(train_set) // args.batch_size) * args.max_epochs

    pool = ActivePool(train_set, batch_size=args.batch_size)
    init_sampler = RandomSampling(None, pool, int(len(train_set)*args.initial_label_rate))
    sampler = get_sampler(args.query_type)(None, pool, size=args.query_size, device=device)

    init_samples = init_sampler()
    pool.update(init_samples)

    summaries = []

    for stage in range(num_stages):

        logger.info(f"Start stage {stage}")
        run_summary = {"stage": stage}
        
        labeled_dl = pool.get_labeled_dataloader()

        model = get_model(num_classes=num_classes).to(device)
        sampler.update_model(model)
        wandb.watch(model, log='all')

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=0.01)

        epochs = 0
        steps = 0
        training = True

        pbar = tqdm(total=max_steps, desc="Train on labeled pool", unit="step")
        while training:
            all_targets = []
            all_preds   = []

            model.train()
            for X, y in labeled_dl:
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
                pbar.update(1)
                if steps > max_steps:
                    training = False
                    break
            
            acc = accuracy_score(all_targets, all_preds)

            epochs += 1
            if acc > args.early_stopping_threshold:
                logger.info(f"Early stopped at epoch {epochs+1} with accuracy score {acc:.3f}")
                break
        pbar.close()
        
        train_stats = {"accuracy": acc, "epochs": epochs, "steps": steps}
        train_stats = log_metrics("train", train_stats)
        run_summary.update(train_stats)
        
        all_targets = []
        all_preds = []

        model.eval()
        with torch.no_grad():
            for X, y in tqdm(test_dl, desc="Evaluate on test set", unit="batch"):
                X = X.to(device)
                y = y.to(device)

                out = model(X)
                loss = criterion(out, y)

                pred = torch.argmax(out, dim=1)
                all_preds.extend(pred.tolist())
                all_targets.extend(y.detach().cpu().tolist())
        
        acc = accuracy_score(all_targets, all_preds)
        
        test_stats = {"accuracy": acc}
        test_stats = log_metrics("test", test_stats)
        run_summary.update(test_stats)

        logger.info("Query on unlabeled pool")
        result = sampler()
        query_stats = {
            "length": len(result.indices), 
            "time": result.info["time"], 
            "target_ids": result.indices, 
        }
        
        # Analyze the query
        # TODO: draw histogram of actual gradients + expected gradients!
        unlabeled_targets = pool.get_unlabeled_targets()
        unlabeled_targets = np.asarray(unlabeled_targets)
        query_targets     = unlabeled_targets[result.indices]

        # Draw a histogram of the query samples
        bins = np.arange(0, num_classes+1) if num_classes < wandb.Histogram.MAX_LENGTH else wandb.Histogram.MAX_LENGTH

        _, cnt = np.unique(query_targets, return_counts=True)
        freq = cnt / np.sum(cnt)
        query_stats.update({
            "hist_targets": wandb.Histogram(np_histogram=np.histogram(query_targets, bins=bins)),
            "hist_scores":  wandb.Histogram(np_histogram=np.histogram(result.scores)),
            "entropy": calc_entropy(freq),
        })
        query_stats = log_metrics("query", query_stats)
        run_summary.update(query_stats)

        pool.update(result)
        logging.info(f"Labeled pool updated with size {len(result.indices)}, time consumed {result.info['time']:.1f}s")
        logging.info(pool)
        
        # not possible to log wandb objects in json
        ignore_list = ["query/hist_targets", "query/hist_scores"]
        json_summary = {k: v for k, v in run_summary.items() if k not in ignore_list}
        summaries.append(json_summary)

    summary_df = pd.DataFrame(summaries)
    save_file_name = f"{args.run_name}_{args.query_type}_{args.dataset}_summary.csv"
    summary_df.to_csv(save_file_name, encoding="utf-8", index=False)
    print(f"Saved all results in {save_file_name}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int)

    parser.add_argument('--run_name', type=str, default="exp")
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--save_path', type=str, default="./saved")
    parser.add_argument('--logging_path', type=str, default="./logging")
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
        config=args,
    )

    main(args)
