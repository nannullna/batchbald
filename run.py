from email.policy import default
from typing import List, Dict, Any, Optional
from collections import defaultdict
import os
import sys
import json
import logging
import datetime
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
from methods import BALD, ActiveQuery, BatchBALD, EntropySampling, GeometricMeanSampling, GradientSampling, KMeansSampling, RandomSampling, UncertaintySampling, MarginSampling
from models import MNISTCNN
from utils import QueryResult, set_all_seeds

# Global logger settings
logger = logging.getLogger()
logger.setLevel(logging.INFO)

mean = [0.4915, 0.4823, 0.4468]
std  = [0.2470, 0.2435, 0.2616]

inv_mean = [-mean[i]/std[i] for i in range(3)]
inv_std  = [1.0/std[i] for i in range(3)]

def load_dataset(name: str, eval_ratio: float = 0.0):

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
        train_set = CIFAR100(root="/opt/datasets/cifar100", train=True,  transform=T.Compose([normalize, train_augment]), download=True)
        query_set = CIFAR100(root="/opt/datasets/cifar100", train=True,  transform=normalize, download=True)
        test_set  = CIFAR100(root="/opt/datasets/cifar100", train=False, transform=normalize, download=True)
    
    else:
        raise ValueError("Not a proper dataset name")

    if eval_ratio != 0.0:
        train_ids = np.random.choice(len(train_set), int(len(train_set)*(1.0-eval_ratio)), replace=False)
        eval_ids = list(set(range(len(train_set))) - set(train_ids))

        _train_set = Subset(train_set, indices=train_ids)
        _eval_set  = Subset(query_set, indices=eval_ids)

        logger.info(f"length of train set {len(_train_set)}, test set {len(test_set)}, eval set {len(_eval_set)}")
        return _train_set, test_set, query_set, _eval_set

    else:
        logger.info(f"length of train set {len(train_set)}, test set {len(test_set)}, eval set 0")
        logger.warning("Use entire train_set w/o augmentations to evaluate the model!")
        return train_set, test_set, query_set, None

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
    elif name.lower() == "kmeans":
        return KMeansSampling

def get_optimizer(args, model):
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
    if args.optimizer_type.lower() == "sgd":
        optimizer = optim.SGD(
            optimizer_grouped_parameters, 
            lr=args.learning_rate, 
            momentum=args.momentum,
        )
    elif args.optimizer_type.lower() == "adam":
        optimizer = optim.Adam(
            optimizer_grouped_parameters,
            lr=args.learning_rate, 
            betas=(args.adam_beta1, args.adam_beta2),
        )
    elif args.optimizer_type.loweR() == "adamw":
        optimizer = optim.AdamW(
            optimizer_grouped_parameters, 
            lr=args.learning_rate, 
            betas=(args.adam_beta1, args.adam_beta2),
        )
    return optimizer

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


def train(model, dataloader, criterion, optimizer, scheduler, device, prev_steps: int = 0):
    all_targets = []
    all_preds   = []
    all_losses  = []

    steps = prev_steps

    model.train()
    for X, y in dataloader:
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
        scheduler.step()
    
    acc = accuracy_score(all_targets, all_preds)
    f1  = f1_score(all_targets, all_preds, average="weighted")
    loss = np.mean(all_losses)

    metrics = {"loss": loss, "accuracy": acc, "f1": f1}
    return metrics, steps

def eval(model, dataloader, criterion, device):
    all_targets = []
    all_preds   = []
    all_losses  = []

    model.eval()
    with torch.no_grad():
        for X, y in tqdm(dataloader):
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

    metrics = {"loss": loss, "accuracy": acc, "f1": f1}
    return metrics


def main(args):

    current_time = datetime.datetime.now().strftime("%y%m%d_%H%M")
    args.logging_path = os.path.join(args.logging_path, f"{args.run_name}_{current_time}")
    args.save_path = os.path.join(args.save_path, f"{args.run_name}_{current_time}")

    if args.seed is not None:
        set_all_seeds(args.seed)

    if not os.path.isdir(args.logging_path):
        os.makedirs(args.logging_path)
        logger.warning(f"Logging path {os.path.abspath(args.logging_path)} has been created!")
    
    if not os.path.isdir(args.save_path):
        os.makedirs(args.save_path)
        logger.warning(f"Model save path {os.path.abspath(args.save_path)} has been created!")

    log_file_path = os.path.join(args.logging_path, f"{args.run_name}_{args.query_type}_{args.dataset}.log")

    stream_handler = logging.StreamHandler(sys.stdout)
    file_handler = logging.FileHandler(filename=log_file_path)
    formatter = logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s")

    stream_handler.setLevel(logging.INFO)
    file_handler.setLevel(logging.DEBUG)

    stream_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu") 

    train_set, test_set, query_set, eval_set = load_dataset(args.dataset, eval_ratio=args.eval_ratio)

    # Since the total count of weight updates depends on the number of mini-batches,
    # the training will continue/stop based on the total number of steps with the full train set.
    num_classes = len(train_set.dataset.classes) if isinstance(train_set, Subset) else len(train_set.classes)
    num_stages  = int(len(train_set) * (1-args.initial_label_ratio)) // args.query_size
    max_steps   = (len(train_set) // args.batch_size) * args.max_epochs
    logger.info(f"num_classes: {num_classes}, num_stages: {num_stages}, max_steps: {max_steps}")

    model = get_model(num_classes=num_classes).to(device)
    model.init_weights()
    wandb.watch(model, log='all')

    pool = ActivePool(train_set, batch_size=args.batch_size, query_set=query_set)
    init_sampler = RandomSampling(None, pool, int(len(train_set)*args.initial_label_ratio))
    sampler = get_sampler(args.query_type)(model, pool, size=args.query_size, device=device)

    init_samples = init_sampler()
    logger.info(f"initial labeled samples: {pool.convert_to_original_ids(init_samples.indices)}")

    pool.update(init_samples)
    logger.info(pool)

    eval_dl = DataLoader(eval_set, batch_size=args.batch_size, shuffle=False) if eval_set is not None else None
    test_dl = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    summaries = []

    # ------------------------------------------------------------------- #
    # [Stage]
    # The proceduer of each stage can be described as below:
    #   1) Initialize (or re-initialize) the model's weights & optimizer
    #   2) Train the model on the labeled train set
    #   3) Evaluate the model on the labeled eval set
    #   4) Determine whether to early-stop training
    #   5) Test the model on the test set (for the experimental purpose)
    #   6) Update the labeled & unlabeled pools
    # ------------------------------------------------------------------- #

    for stage in range(num_stages):

        logger.info(f"Start stage {stage}")
        run_summary = {"stage": stage}
        
        num_acquired_points = len(pool.get_labeled_ids())
        labeled_dl = pool.get_labeled_dataloader()

        # simply re-initialize weights to continuously track the model's parameters and gradients
        model.init_weights()
        logger.info("Model re-initialized")
        # sampler.update_model(model)
        # call update_model() only when a new instance of the model is instantiated.

        criterion = nn.CrossEntropyLoss()
        optimizer = get_optimizer(args, model)
        # No weight decay for bias and LayerNorm
        
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
        # reset optimizer for emptying its state dict

        epochs = 0
        steps = 0
        training = True

        best_eval_acc = 0.0
        tolerance = args.tolerance

        pbar = tqdm(total=max_steps, desc="Train on labeled pool", unit="step")
        while training:
            
            # ------------------------------------- #
            # Train
            # ------------------------------------- #
            epochs += 1
            train_metrics, steps = train(model, labeled_dl, criterion, optimizer, scheduler, device, steps)
            pbar.update(steps - pbar.n)
            
            if epochs % args.log_every == 0:
                train_metrics.update({"epochs": epochs})
                log_metrics("training", train_metrics, global_stage=stage)

            # ------------------------------------- #
            # Eval
            # ------------------------------------- #
            if epochs % args.eval_every == 0:
                eval_metrics = eval(model, eval_dl or labeled_dl, criterion, device)
                eval_metrics.update({"epochs": epochs})
                log_metrics("eval", eval_metrics, global_stage=stage, num_acquired=num_acquired_points)

                if eval_metrics["accuracy"] >= best_eval_acc:
                    best_eval_acc = eval_metrics["accuracy"]
                    if tolerance < args.tolerance:
                        logger.info(f"Tolerance reset from {tolerance} to {args.tolerance}")
                    tolerance = args.tolerance
                else:
                    tolerance -= 1
                    logger.info(f"Tolerance reduced to {tolerance}")

            # ------------------------------------- #
            # Early Stopping
            # ------------------------------------- #
            if tolerance < 0:
                logger.info(f"Early stopped at epoch {epochs} after waiting {args.tolerance * args.eval_every} epochs.")
                training = False
                break

            if train_metrics['accuracy'] > args.early_stopping_threshold:
                logger.info(f"Early stopped at epoch {epochs} surpassing the threshold {args.early_stopping_threshold}.")
                training = False
                break

            if steps > max_steps:
                logger.info(f"Stopped as it reached the max steps {max_steps} (current steps: {steps}).")
                training = False
                break
        
        pbar.close()
        logger.info(f"Train results -- Accuracy: train {train_metrics['accuracy']:.3f}, eval {eval_metrics['accuracy']:.3f}")
        
        train_metrics = log_metrics("train", train_metrics, global_stage=stage, num_acquired=num_acquired_points)
        run_summary.update(train_metrics)

        # ------------------------------------- #
        # Test
        # ------------------------------------- #
        test_metrics = eval(model, test_dl, criterion, device)
        test_metrics = log_metrics("test", test_metrics, global_stage=stage, num_acquired=num_acquired_points)
        run_summary.update(test_metrics)

        # ------------------------------------- #
        # Query
        # ------------------------------------- #
        logger.info("Query on unlabeled pool")
        result = sampler()

        unlabeled_targets = np.asarray(pool.get_unlabeled_targets())
        
        query_targets     = unlabeled_targets[result.indices]

        query_metrics = {
            "length": len(result.indices), 
            "time": result.info["time"], 
            "target_ids": result.indices,
            "original_ids": pool.convert_to_original_ids(result.indices), 
        }
        
        # ------------------------------------- #
        # Draw sampled images
        # ------------------------------------- #
        inv_transform = T.Compose([
            T.Normalize(mean=inv_mean, std=inv_std),
            T.ToPILImage(),
        ])

        imgs = []
        lbls = []
        all_classes = pool.get_classes()
        for idx in pool.convert_to_original_ids(result.indices[:50]):
            img, lbl = query_set[idx]
            imgs.append(inv_transform(img))
            lbls.append(all_classes[lbl])
        
        # Draw a histogram of the query samples
        bins = np.arange(0, num_classes+1) if num_classes < wandb.Histogram.MAX_LENGTH else wandb.Histogram.MAX_LENGTH

        _, cnt = np.unique(query_targets, return_counts=True)
        freq = cnt / np.sum(cnt)
        query_metrics.update({
            "query_images": [wandb.Image(img, caption=lbl) for img, lbl in zip(imgs, lbls)],
            "hist_targets": wandb.Histogram(np_histogram=np.histogram(query_targets, bins=bins)),
            "hist_scores":  wandb.Histogram(np_histogram=np.histogram(result.scores)),
            "entropy": calc_entropy(freq),
        })
        query_metrics = log_metrics("query", query_metrics, global_stage=stage, num_acquired=num_acquired_points)
        run_summary.update(query_metrics)

        # ------------------------------------- #
        # Update labeled & unlabeled sets
        # ------------------------------------- #
        pool.update(result)
        logging.info(f"Labeled pool updated with size {len(result.indices)}, time consumed {result.info['time']:.1f}s")
        logging.info(pool)

        # sanity check
        assert len(set(pool.get_labeled_ids() + pool.get_unlabeled_ids())) == len(train_set)
        
        # not possible to log wandb objects in json
        ignore_list = ["query/hist_targets", "query/hist_scores", "query/query_images"]
        json_summary = {k: v for k, v in run_summary.items() if k not in ignore_list}
        summaries.append(json_summary)

    # ------------------------------------- #
    # Summarize the experiment results
    # ------------------------------------- #
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
    parser.add_argument('--log_every', type=int, default=50)
    parser.add_argument('--eval_every', type=int, default=1)
    parser.add_argument('--device', type=str, default="cuda:0")

    parser.add_argument('--max_epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--early_stopping_threshold', type=float, default=0.9)

    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--optimizer_type', type=str, choices=["sgd", "adam", "adamw"], default="sgd")
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--adam_beta1', type=float, default=0.9)
    parser.add_argument('--adam_beta2', type=float, default=0.999)

    parser.add_argument('--query_size', type=int, default=100)
    parser.add_argument('--query_type', type=str, default="random")
    parser.add_argument('--initial_label_ratio', type=float, default=0.1)
    parser.add_argument('--eval_ratio', type=float, default=0.1)
    parser.add_argument('--exclude_ratio', type=float, default=0.0)
    parser.add_argument('--tolerance', type=int, default=0)

    parser.add_argument('--wandb_project', type=str, default="active_learning")

    parser.add_argument('-v', '--verbose', action='store_true')
    
    args = parser.parse_args()

    wandb.init(
        project=args.wandb_project,
        name=f"{args.run_name}_{args.query_type}_{args.dataset}",
        config=args,
    )

    main(args)
