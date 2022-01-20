from abc import abstractmethod
from typing import Optional, Union, List, Dict
from dataclasses import dataclass, field
import gc
import time

import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset

from pool import ActivePool
from utils import QueryResult

class ActiveQuery:
    """An abstract class for deep active learning"""
    def __init__(self, model: nn.Module, pool: ActivePool, size: int=1, device: torch.device=None):
        self.model = model
        self.pool = pool
        self.size = size
        self.device = device

    def __repr__(self):
        return f"[{self.__class__.__name__}]"

    def __call__(self, size: Optional[int]=None, **kwargs) -> QueryResult:
        return self.query(size)
        
    def query(self, size: Optional[int]=None, **kwargs) ->QueryResult:
        """This function wraps `_query_impl()` method and provides additional functionalities,
        including logging information regrading query method, time consumed, and etc.
        You can override this method and implement custom behaviors."""
        size = size or self.size
        remainings = len(self.pool.unlabeled_data)
        start = time.time()
        
        if remainings == 0 or size == 0:
            # nothing left to query
            result = QueryResult(scores=[], indices=[])
        elif remainings < size:
            # query all the remainings
            result = QueryResult(
                scores=[1] * remainings, 
                indices=list(range(remainings)),
            )
        else:
            # its behavior depends on actual implementation!
            result = self._query_impl(size)

        end = time.time()
        result.info.update({"method": self.__class__.__name__, "time": end-start})
        return result

    @abstractmethod
    def _query_impl(self, size: int, **kwargs) -> QueryResult:
        pass
        
    def update_model(self, model: nn.Module):
        del self.model
        gc.collect()
        self.model = model


class RandomSampling(ActiveQuery):
    def _query_impl(self, size: int) -> QueryResult:
        return QueryResult(
            scores=[1] * size,
            indices=np.random.choice(np.arange(len(self.pool.unlabeled_data)), size, replace=False).tolist(),
        )

class UncertaintySampling(ActiveQuery):
    def _query_impl(self, size: int, **kwargs) -> QueryResult:
        dataloader = self.pool.get_unlabeled_dataloader(shuffle=False)
        all_scores = []

        device = self.device if self.device is not None else torch.device("cuda")
        with torch.no_grad():
            for X, _ in dataloader:
                X = X.to(device)
                # y = y.to(device)
                out = self.model(X)
                score, pred = torch.max(out, dim=1)
                all_scores.extend(score.detach().cpu().tolist())

        all_scores = np.array(all_scores)
        indices = all_scores.argsort()[:size].tolist()
        scores = np.sort(all_scores)[:size].tolist()

        return QueryResult(
            indices=indices,
            scores=scores,
        )

class MarginSampling(ActiveQuery):
    def _query_impl(self, size: int, **kwargs) -> QueryResult:
        dataloader = self.pool.get_unlabeled_dataloader(shuffle=False)
        all_scores = []

        device = self.device if self.device is not None else torch.device("cuda")
        with torch.no_grad():
            for X, _ in dataloader:
                X = X.to(device)
                # y = y.to(device)
                out = self.model(X)
                prob = torch.softmax(out, dim=1)
                val, ids = torch.topk(prob, k=2, dim=1)
                score = val[:, 0] - val[:, 1]
                pred = ids[:, 0]
                all_scores.extend(score.detach().cpu().tolist())

        all_scores = np.array(all_scores)
        indices = all_scores.argsort()[:size].tolist()
        scores = np.sort(all_scores)[:size].tolist()

        return QueryResult(
            indices=indices,
            scores=scores,
        )