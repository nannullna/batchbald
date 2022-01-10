from typing import List
from dataclasses import dataclass

import numpy as np

import torch


@dataclass
class CandidateBatch:
    scores: List[float]
    indices: List[int]


def get_mixture_prob_dist(p1, p2, m):
    """Generate fake inputs for simulating different outputs per model"""
    return (1.0 - m) * np.asarray(p1) + m * np.asarray(p2)


def nested_to_tensor(l):
    return torch.stack(list(map(torch.as_tensor, l)))
