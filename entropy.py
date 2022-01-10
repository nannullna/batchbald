import math

import torch
import torch.nn as nn


def calculate_conditional_entropy(log_probs: torch.Tensor, max_N: int = 1024) -> torch.Tensor:
    """Returns the entropies.
    
    Args:
        log_probs (torch.Tensor): tensor of shape (N, K, C) where N = # data, K = # samples, C = # classes
        max_N (int, optional): the maximum size of a single-batched calculation

    Returns:
        torch.DoubleTensor of entropies
    """
    if log_probs.ndim != 3:
        raise ValueError("The input must be in shape (N, K, C).")

    N, K, C = log_probs.shape
    steps = N // max_N if N % max_N == 0 else N // max_N + 1

    entropies = torch.empty((N,), dtype=torch.double, device=log_probs.device)

    for step in range(steps):
        start, end = max_N * step, min(max_N * (step + 1), N)
        _log_probs = log_probs[start:end]

        _entropy = _log_probs * torch.exp(_log_probs)
        _entropy = -torch.sum(_entropy, dim=(1, 2)) / K  # squeeze K and C dimension
        entropies[start:end].copy_(_entropy)
        
    return entropies


def calculate_entropy(log_probs: torch.Tensor, max_N: int = 1024) -> torch.Tensor:
    """Returns the entropies.

    Args:
        log_probs (torch.Tensor): tensor of shape (N, K, C) where N = # data, K = # samples, C = # classes
        max_N (int, optional): the maximum size of a single-batched calculation

    Returns:
        torch.DoubleTensor of entropies
    """
    if log_probs.ndim != 3:
        raise ValueError("The input must be in shape (N, K, C).")

    N, K, C = log_probs.shape

    entropies = torch.empty((N,), dtype=torch.double, device=log_probs.device)
    steps = N // max_N if N % max_N == 0 else N // max_N + 1

    for step in range(steps):
        start, end = max_N * step, min(max_N * (step + 1), N)
        _log_probs = log_probs[start:end]

        # average over K dimension to calculate p_hat
        _mean_log_probs = torch.logsumexp(_log_probs, dim=1) - math.log(K)
        
        _entropy = _mean_log_probs * torch.exp(_mean_log_probs)
        _entropy = -torch.sum(_entropy, dim=1)  # squeeze C dimension
        entropies[start:end].copy_(_entropy)

    return entropies
