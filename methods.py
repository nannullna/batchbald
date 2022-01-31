from abc import abstractmethod
from typing import Optional, Sequence, Union, List, Dict
from dataclasses import dataclass, field
import gc
import math
import time

import numpy as np
from sklearn import cluster

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset

from tqdm.auto import tqdm

from pool import ActivePool
from utils import QueryResult

class ActiveQuery:
    """An abstract class for deep active learning"""
    def __init__(self, model: nn.Module, pool: ActivePool, size: int=1, device: torch.device=None, **kwargs):
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
    def get_query_from_scores(scores: Union[Sequence[float], Sequence[int]], size: int, higher_is_better: bool = False, return_list: bool=False):
        scores  = np.asarray(scores, dtype=np.float32)
        indices = scores.argsort()[-size:] if higher_is_better else scores.argsort()[:size]
        scores  = np.sort(scores)[-size:]  if higher_is_better else np.sort(scores)[:size]
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

        device = self.device or torch.device("cuda")

        self.model.eval()
        with torch.no_grad():
            for X, _ in tqdm(dataloader, desc="Query by smallest maximum probabilities"):
                X = X.to(device)
                # y = y.to(device)
                out = self.model(X)

                # Actually, it is not required to apply softmax...
                # but it is due to the analysis purpose.
                prob = torch.softmax(out, dim=1)
                score, pred = torch.max(prob, dim=1)
                all_scores.extend(score.detach().cpu().tolist())

        # returns the query with the k least confident probability
        return self.get_query_from_scores(all_scores, size=size, higher_is_better=False)

class MarginSampling(ActiveQuery):
    def _query_impl(self, size: int, **kwargs) -> QueryResult:
        dataloader = self.pool.get_unlabeled_dataloader(shuffle=False)
        all_scores = []

        device = self.device or torch.device("cuda")

        self.model.eval()
        with torch.no_grad():
            for X, _ in tqdm(dataloader, desc="Query by smallest margins"):
                X = X.to(device)
                # y = y.to(device)
                out = self.model(X)
                prob = torch.softmax(out, dim=1)
                val, ids = torch.topk(prob, k=2, dim=1)
                score = val[:, 0] - val[:, 1]
                pred = ids[:, 0]
                all_scores.extend(score.detach().cpu().tolist())

        # returns the query with the k smallest margins
        return self.get_query_from_scores(all_scores, size=size, higher_is_better=False)

class EntropySampling(ActiveQuery):

    @staticmethod
    def calc_entropy(x: torch.Tensor, log_p: bool=False, keepdim: bool=False) -> torch.Tensor:
        if log_p:
            entry = torch.exp(x) * x
        else:
            entry = x * torch.log(x)
            entry[x == 0.0] = 0.0
        entropy = -torch.sum(entry, dim=1, keepdim=keepdim)
        return entropy

    def _query_impl(self, size: int, **kwargs) -> QueryResult:
        dataloader = self.pool.get_unlabeled_dataloader(shuffle=False)
        all_scores = []

        device = self.device or torch.device("cuda")

        self.model.eval()
        with torch.no_grad():
            for X, _ in tqdm(dataloader, desc="Query by largest entropies"):
                X = X.to(device)
                # y = y.to(device)
                out = self.model(X)
                log_prob = torch.log_softmax(out, dim=1)
                score = self.calc_entropy(log_prob, log_p=True)
                all_scores.extend(score.detach().cpu().tolist())
        
        # returns the query with the k highest entropies
        return self.get_query_from_scores(all_scores, size=size, higher_is_better=True)

class GeometricMeanSampling(ActiveQuery):
    def _query_impl(self, size: int, **kwargs) -> QueryResult:
        dataloader = self.pool.get_unlabeled_dataloader(shuffle=False)
        all_scores = []

        device = self.device or torch.device("cuda")

        self.model.eval()
        with torch.no_grad():
            for X, _ in tqdm(dataloader, desc="Query by largest geometric means"):
                X = X.to(device)
                # y = y.to(device)
                out = self.model(X)
                log_prob = torch.log_softmax(out, dim=1)
                score = torch.exp(torch.mean(log_prob, dim=1))
                all_scores.extend(score.detach().cpu().tolist())

        return self.get_query_from_scores(all_scores, size=size, higher_is_better=True)


class BALD(ActiveQuery):

    def __init__(self, model: nn.Module, pool: ActivePool, size: int = 1, device: torch.device = None, **kwargs):
        super().__init__(model, pool, size, device, **kwargs)
        if "num_samples" in kwargs:
            self.K = kwargs["num_samples"]
        elif "K" in kwargs:
            self.K = kwargs["K"]
        else:
            self.K = 10

        if self.model is not None:
            self.model.global_pool.register_forward_hook(lambda m, i, out: torch.dropout(out, p=0.5, train=True))

    def update_model(self, model: nn.Module):
        super().update_model(model)
        if self.model is not None:
            self.model.global_pool.register_forward_hook(lambda m, i, out: torch.dropout(out, p=0.5, train=True))

    @staticmethod
    def calc_entropy(x: torch.Tensor, log_p: bool = False, keepdim: bool = False) -> torch.Tensor:
        # reduce dimension
        if log_p:
            mean_log_prob = torch.logsumexp(x, dim=1) - math.log(x.size(1))
            entry = torch.exp(mean_log_prob) * mean_log_prob
        else:
            mean_prob = torch.mean(x, dim=1)
            entry = mean_prob * torch.log(mean_prob)
            entry[mean_prob == 0.0] = 0.0
        entropy = -torch.sum(entry, dim=1, keepdim=keepdim)
        return entropy

    @staticmethod
    def calc_conditional_entropy(x: torch.Tensor, log_p: bool = False, keepdim: bool = False) -> torch.Tensor:
        if log_p:
            entry = torch.exp(x) * x
        else:
            entry = x * torch.log(x)
            entry[x == 0.0] = 0.0
        entropy = -torch.sum(entry, dim=-1)
        entropy = torch.mean(entropy, dim=1, keepdim=keepdim)
        return entropy


    def _query_impl(self, size: int, **kwargs) -> QueryResult:

        _batch_size = self.pool.batch_size//self.K
        _batch_size = 2**(math.ceil(math.log(_batch_size, 2)))

        dataloader = self.pool.get_unlabeled_dataloader(batch_size=_batch_size)
        all_scores = []

        device = self.device or torch.device("cuda")

        self.model.eval()
        with torch.no_grad():
            for X, _ in tqdm(dataloader, desc="Query by mutual information"):
                B = X.size(0)
                in_shape = list(X.shape)[1:]

                # create batch input
                X = X.to(device).unsqueeze(1) # [B, ...]
                X = X.expand([-1, self.K] + in_shape).contiguous() # [B, K, ...]
                X = X.view([B*self.K] + in_shape) # [B*K, ...]

                out = self.model(X)
                out_shape = list(out.shape)[1:]
                log_prob = torch.log_softmax(out, dim=1)
                log_prob = log_prob.view([B, self.K] + out_shape)

                score = self.calc_entropy(log_prob, log_p=True) - self.calc_conditional_entropy(log_prob, log_p=True)
                all_scores.extend(score.detach().cpu().tolist())

        return self.get_query_from_scores(all_scores, size=size, higher_is_better=True)


# TODO
class BatchBALD(ActiveQuery):
    pass

# TODO
class GradientSampling(ActiveQuery):
    pass

# TODO
class AdaptiveGradientSampling(ActiveQuery):
    pass

class KMeansSampling(ActiveQuery):

    from sklearn.cluster import KMeans

    def get_embeddings(self) -> torch.Tensor:
        embeddings = []
        handle = self.model.global_pool.register_forward_hook(lambda m, i, o: embeddings.append(o.flatten(start_dim=1)))
        self.model.eval()
        with torch.no_grad():
            for X, _ in tqdm(self.pool.get_unlabeled_dataloader(), desc="Get embeddings"):
                X = X.to(self.device)
                self.model(X)
        # prevent memory leaks
        handle.remove()
        return torch.cat(embeddings, dim=0)

    def _query_impl(self, size: int, **kwargs) -> QueryResult:
        embs = self.get_embeddings()
        embs_arr = embs.detach().cpu().numpy()
        kmeans = cluster.KMeans(n_clusters=size)
        kmeans.fit(embs_arr)

        cluster_ids = kmeans.predict(embs_arr)
        centroids   = kmeans.cluster_centers_[cluster_ids]
        distances   = (embs_arr - centroids) ** 2
        distances   = np.sqrt(distances.sum(axis=1))

        # prevent memory leaks
        torch.cuda.empty_cache()
        query_ids = [np.arange(embs_arr.shape[0])[cluster_ids==i][distances[cluster_ids==i].argmin()] for i in range(size)]
        
        return QueryResult(
            indices=query_ids, 
            scores=distances[query_ids].tolist(),
        )
        