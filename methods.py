from abc import abstractmethod
from typing import Optional, Union, List, Dict
from dataclasses import dataclass, field
import gc

import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset

from pool import ActivePool
from utils import QueryResult

class ActiveQuery:
    """An abstract class for deep active learning"""
    def __init__(self, model: nn.Module, pool: ActivePool, size: int=1):
        self.model = model
        self.pool = pool
        self.size = size

    def __repr__(self):
        return f"[{self.__class__.__name__}]"

    def __call__(self, size: Optional[int]=None) -> QueryResult:
        size = size or self.size
        remainings = len(self.pool.unlabeled_data)
        if remainings == 0:
            return QueryResult(scores=[], indices=[])
        elif remainings < size:
            return QueryResult(
                scores=[1] * remainings, 
                indices=list(range(remainings)),
            )
        else:
            return self.query(size)

    @abstractmethod
    def query(self, size: int) -> QueryResult:
        pass
        
    def update_model(self, model: nn.Module):
        del self.model
        gc.collect()
        self.model = model


class RandomSampling(ActiveQuery):
    def query(self, size: int) -> QueryResult:
        return QueryResult(
            scores=[1] * size,
            indices=np.random.choice(np.arange(len(self.pool.unlabeled_data)), size, replace=False).tolist(),
        )