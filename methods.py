from abc import abstractmethod
from typing import Optional, Sequence, Union, List, Dict
from dataclasses import dataclass, field
import gc
import math
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

    @staticmethod
    def get_query_from_scores(scores: Union[Sequence[float], Sequence[int]], size: int, return_list: bool=False):
        scores  = np.asarray(scores, dtype=np.float32)
        indices = scores.argsort()[:size]
        scores  = np.sort(scores)[:size]
        if return_list:
            return QueryResult(indices=indices.tolist(), scores=scores.tolist())
        else:
            return QueryResult(indices=indices, scores=scores)


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

        self.model.eval()
        with torch.no_grad():
            for X, _ in dataloader:
                X = X.to(device)
                # y = y.to(device)
                out = self.model(X)
                score, pred = torch.max(out, dim=1)
                all_scores.extend(score.detach().cpu().tolist())

        return self.get_query_from_scores(all_scores, size=size)

class MarginSampling(ActiveQuery):
    def _query_impl(self, size: int, **kwargs) -> QueryResult:
        dataloader = self.pool.get_unlabeled_dataloader(shuffle=False)
        all_scores = []

        device = self.device if self.device is not None else torch.device("cuda")

        self.model.eval()
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

        return self.get_query_from_scores(all_scores, size=size)

class EntropySampling(ActiveQuery):

    @staticmethod
    def calc_entropy(x: torch.Tensor, log_p: bool=False, keepdim: bool=False) -> torch.Tensor:
        if log_p:
            entry = torch.exp(x) * x
        else:
            entry = x * torch.log(x)
            entry[x == 0.0] = 0.0
        entropy = -torch.sum(x, dim=1, keepdim=keepdim)
        return entropy

    def _query_impl(self, size: int, **kwargs) -> QueryResult:
        dataloader = self.pool.get_unlabeled_dataloader(shuffle=False)
        all_scores = []

        device = self.device if self.device is not None else torch.device("cuda")

        self.model.eval()
        with torch.no_grad():
            for X, _ in dataloader:
                X = X.to(device)
                # y = y.to(device)
                out = self.model(X)
                logits = torch.log_softmax(out, dim=1)
                score = self.calc_entropy(logits, log_p=True)
                all_scores.extend(score.detach().cpu().tolist())
        
        return self.get_query_from_scores(all_scores, size=size)

class BALD(ActiveQuery):

    def __init__(self, model: nn.Module, pool: ActivePool, num_samples: int = 10, size: int = 1, device: torch.device = None):
        super().__init__(model, pool, size, device)
        if isinstance(model, nn.Module):
            # safety check here!
            self.model.global_pool.register_forward_hook(lambda m, i, out: torch.dropout(out, p=0.5, training=True))
        self.K = num_samples

    def update_model(self, model: nn.Module):
        del self.model
        gc.collect()
        self.model = model
        # add dropout right before the final layer
        self.model.global_pool.register_forward_hook(lambda m, i, out: torch.dropout(out, p=0.5, train=True))

    @staticmethod
    def calc_entropy(x: torch.Tensor, log_p: bool = False, keepdim: bool = False) -> torch.Tensor:
        # reduce dimension
        if log_p:
            mean_log_prob = torch.logsumexp(x, dim=1) - torch.log(torch.tensor(x.size(1)))
            entry = torch.exp(mean_log_prob) * mean_log_prob
        else:
            mean_prob = torch.mean(x, dim=1)
            entry = mean_prob * torch.log(mean_prob)
            entry[mean_prob == 0.0] = 0.0
        entropy = -torch.sum(x, dim=1, keepdim=keepdim)
        return entropy

    @staticmethod
    def calc_conditional_entropy(x: torch.Tensor, log_p: bool = False, keepdim: bool = False) -> torch.Tensor:
        if log_p:
            entry = torch.exp(x) * x
        else:
            entry = x * torch.log(x)
            entry[x == 0.0] = 0.0
        entropy = torch.sum(entry, dim=-1)
        entropy = torch.mean(entropy, dim=1, keepdim=keepdim)
        return entropy


    def _query_impl(self, size: int, **kwargs) -> QueryResult:

        _batch_size = self.pool.batch_size//self.K
        _batch_size = 2**(math.ceil(math.log(_batch_size, 2)))

        dataloader = self.pool.get_unlabeled_dataloader(batch_size=_batch_size)
        all_scores = []

        device = self.device if self.device is not None else torch.device("cuda")

        self.model.train()
        with torch.no_grad():
            for X, _ in dataloader:
                B = X.size(0)
                ndim = X.ndim

                # create batch input
                X = X.to(device).unsqueeze(1) # [B, ...]
                X = X.expand([-1, self.K] + [-1] * (ndim-1)).contiguous() # [B, K, ...]
                X = X.view([B*self.K] + [-1]*(ndim-1)) # [B*K, ...]

                out = self.model(X)
                logits = torch.log_softmax(out, dim=1)
                logits = logits.view([B, self.K] + [-1]*(ndim-1))

                score = self.calc_entropy(out, log_p=True) - self.calc_conditional_entropy(out, log_p=True)
                all_scores.extend(score.detach().cpu().tolist())

        return self.get_query_from_scores(all_scores, size=size)