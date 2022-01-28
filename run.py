from email.policy import default
from typing import List, Dict, Any, Optional
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
from methods import BALD, ActiveQuery, BatchBALD, EntropySampling, GeometricMeanSampling, GradientSampling, RandomSampling, UncertaintySampling, MarginSampling
from models import MNISTCNN
from utils import QueryResult, set_all_seeds

# Global logger settings
logger = logging.getLogger()
logger.setLevel(logging.INFO)

mean = [0.4915, 0.4823, 0.4468]
std  = [0.2470, 0.2435, 0.2616]

inv_mean = [-mean[i]/std[i] for i in range(3)]
inv_std  = [1.0/std[i] for i in range(3)]

def load_dataset(name: str):

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])
    train_augment= T.Compose([ 
        T.RandomHorizontalFlip(),
        T.RandomRotation((-15, 15)),
        T.RandomCrop(32, padding=4),
    ])

    if name.lower() == "mnist":
        train_set = MNIST(root="/opt/datasets/mnist", train=True,  transform=T.ToTensor(), download=True)
        query_set = train_set
        test_set  = MNIST(root="/opt/datasets/mnist", train=False, transform=T.ToTensor(), download=True)

    elif name.lower() == "cifar10":
        train_set = CIFAR10(root="/opt/datasets/cifar10", train=True,  transform=T.Compose([normalize, train_augment]), download=True)
        query_set = CIFAR10(root="/opt/datasets/cifar10", train=True,  transform=normalize, download=True)
        test_set  = CIFAR10(root="/opt/datasets/cifar10", train=False, transform=normalize, download=True)

    elif name.lower() == "cifar100":
        train_set = CIFAR10(root="/opt/datasets/cifar100", train=True,  transform=T.Compose([normalize, train_augment]), download=True)
        query_set = CIFAR10(root="/opt/datasets/cifar100", train=True,  transform=normalize, download=True)
        test_set  = CIFAR10(root="/opt/datasets/cifar100", train=False, transform=normalize, download=True)
    
    else:
        raise ValueError("Not a proper dataset name")

    logger.info(f"length of train set {len(train_set)}, test set {len(test_set)}")
    return train_set, test_set, query_set

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
    elif name.lower() == "mean":
        return GeometricMeanSampling
    elif name.lower() == "entropy":
        return EntropySampling
    elif name.lower() == "bald":
        return BALD
    elif name.lower() == "batchbald":
        return BatchBALD
    elif name.lower() == "gradient":
        return GradientSampling

def log_metrics(prefix: str, metrics: Dict[str, Any], **kwargs) -> Dict[str, Any]:
    _metrics = {}
    for k in list(metrics.keys()):
        if not k.startswith(f"{prefix}/"):
            _metrics[f"{prefix}/{k}"] = metrics[k]
        else:
            _metrics[k] = metrics[k]
    _metrics.update(kwargs)
    wandb.log(_metrics)
    logger.info(_metrics)
    return _metrics

def calc_entropy(p: np.ndarray):
    v = p * np.log(p)
    v[p == 0.0] = 0.0
    return -np.sum(v)

def main(args):

    stream_handler = logging.StreamHandler(sys.stdout)
    file_handler = logging.FileHandler(filename=f"{args.run_name}_{args.query_type}_{args.dataset}.log")

    stream_handler.setLevel(logging.INFO)
    file_handler.setLevel(logging.DEBUG)

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)

    if args.seed is not None:
        set_all_seeds(args.seed)

    if not os.path.isdir(args.logging_path):
        os.makedirs(args.logging_path)
        logger.warning(f"Logging path {os.path.abspath(args.logging_path)} has been created!")
    
    if not os.path.isdir(args.save_path):
        os.makedirs(args.save_path)
        logger.warning(f"Model save path {os.path.abspath(args.save_path)} has been created!")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu") 

    train_set, test_set, query_set = load_dataset(args.dataset)
    test_dl = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    num_classes = len(train_set.classes)
    num_stages  = int(len(train_set) * (1-args.initial_label_rate)) // args.query_size
    max_steps   = (len(train_set) // args.batch_size) * args.max_epochs
    log_every   = 100
    logger.info(f"num_classes: {num_classes}, num_stages: {num_stages}, max_steps: {max_steps}")

    model = get_model(num_classes=num_classes).to(device)
    model.init_weights()
    wandb.watch(model, log='all')

    pool = ActivePool(train_set, batch_size=args.batch_size)
    init_sampler = RandomSampling(None, pool, int(len(train_set)*args.initial_label_rate))
    sampler = get_sampler(args.query_type)(model, pool, size=args.query_size, device=device)

    init_samples = init_sampler()
    pool.update(init_samples)

    summaries = []

    for stage in range(num_stages):

        logger.info(f"Start stage {stage}")
        run_summary = {"stage": stage}
        
        num_acquired_points = len(pool.get_labeled_ids())
        labeled_dl = pool.get_labeled_dataloader()

        # simply re-initialize weights to continuously track the model's parameters and gradients
        model.init_weights()
        logger.info("Model re-initialized")
        # sampler.update_model(model)

        criterion = nn.CrossEntropyLoss()

        # No weight decay for bias and LayerNorm
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": args.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = optim.SGD(
            optimizer_grouped_parameters, 
            lr=args.learning_rate, 
            momentum=args.momentum, 
            weight_decay=args.weight_decay,
        )
 #       scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
        # reset optimizer for emptying its state dict

        epochs = 0
        steps = 0
        training = True

        pbar = tqdm(total=max_steps, desc="Train on labeled pool", unit="step")
        while training:
            all_targets = []
            all_preds   = []
            all_losses  = []

            model.train()
            for batch_idx, (X, y) in enumerate(labeled_dl):
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
                all_losses.append(loss.item())

                steps += 1
                pbar.update(1)                    
            
            acc = accuracy_score(all_targets, all_preds)
            f1  = f1_score(all_targets, all_preds, average="weighted")
            loss = np.mean(all_losses)
            if (epochs+1) % log_every == 0:
                log_metrics("training", {"loss": loss, "accuracy": acc, "f1": f1, "epoch": epochs}, global_stage=stage)

            epochs += 1
            if acc > args.early_stopping_threshold:
                logger.info(f"Early stopped at epoch {epochs} with accuracy score {acc:.3f}")
                break

            if steps > max_steps:
                logger.info(f"Stopped as it reached the max steps {max_steps} with accuracy score {acc:.3f}")
                training = False
                break

            # scheduler.step()

        pbar.close()
        
        train_stats = {"train/accuracy": acc, "train/f1": f1, "train/epochs": epochs, "train/steps": steps, "train/loss": loss}
        log_metrics("train", train_stats, global_stage=stage, num_acquired=num_acquired_points)
        run_summary.update(train_stats)
        
        all_targets = []
        all_preds = []
        all_losses  = []

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
                all_losses.append(loss.item())
        
        acc = accuracy_score(all_targets, all_preds)
        f1  = f1_score(all_targets, all_preds, average="weighted")
        loss = np.mean(all_losses)
        
        test_stats = {"test/accuracy": acc, "test/f1": f1, "test/loss": loss}
        run_summary.update(test_stats)
        log_metrics("test", test_stats, global_stage=stage, num_acquired=num_acquired_points)

        logger.info("Query on unlabeled pool")
        result = sampler()
        query_stats = {
            "query/length": len(result.indices), 
            "query/time": result.info["time"], 
            "query/target_ids": result.indices, 
        }
        
        # Analyze the query
        # TODO: draw histogram of actual gradients + expected gradients!
        unlabeled_targets = pool.get_unlabeled_targets()
        unlabeled_targets = np.asarray(unlabeled_targets)
        query_targets     = unlabeled_targets[result.indices]

        # TODO: Log query images
        unlabeled_ids = pool.get_unlabeled_ids()
        inv_transform = T.Compose([
            T.Normalize(mean=inv_mean, std=inv_std),
            T.ToPILImage(),
        ])

        imgs = []
        labels = []
        for i in range(32):
            original_idx = unlabeled_ids[result.indices[i]]
            img, lbl = query_set[original_idx]
            imgs.append(inv_transform(img))
            labels.append(train_set.classes[lbl])
        
        # Draw a histogram of the query samples
        bins = np.arange(0, num_classes+1) if num_classes < wandb.Histogram.MAX_LENGTH else wandb.Histogram.MAX_LENGTH

        _, cnt = np.unique(query_targets, return_counts=True)
        freq = cnt / np.sum(cnt)
        query_stats.update({
            "query/query_images": [wandb.Image(img, caption=lbl) for img, lbl in zip(imgs, labels)],
            "query/hist_targets": wandb.Histogram(np_histogram=np.histogram(query_targets, bins=bins)),
            "query/hist_scores":  wandb.Histogram(np_histogram=np.histogram(result.scores)),
            "query/entropy": calc_entropy(freq),
        })
        run_summary.update(query_stats)
        log_metrics("query", query_stats, global_stage=stage, num_acquired=num_acquired_points)

        pool.update(result)
        logging.info(f"Labeled pool updated with size {len(result.indices)}, time consumed {result.info['time']:.1f}s")
        logging.info(pool)
        
        # not possible to log wandb objects in json
        ignore_list = ["query/hist_targets", "query/hist_scores", "query/query_images"]
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

    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=0.001)

    parser.add_argument('--query_size', type=int, default=100)
    parser.add_argument('--query_type', type=str, default="random")
    parser.add_argument('--initial_label_rate', type=float, default=0.1)
    parser.add_argument('--exclude_ratio', type=float, default=0.0)

    parser.add_argument('--wandb_project', type=str, default="active_learning")

    parser.add_argument('-v', '--verbose', action='store_true')
    
    args = parser.parse_args()

    wandb.init(
        project=args.wandb_project,
        name=f"{args.run_name}_{args.query_type}_{args.dataset}",
        config=args,
    )

    main(args)
