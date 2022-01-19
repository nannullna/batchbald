from abc import abstractmethod
from cProfile import label
from turtle import update
from typing import Optional, Union, List, Dict

from torch.utils.data import Dataset, DataLoader, Subset

from utils import QueryResult


class ActivePool:
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int = 1,
        labeled_ids: Optional[List[int]] = None,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        
        labeled_ids = list(labeled_ids) if labeled_ids else []

        self.labeled_data   = Subset(self.dataset, labeled_ids)
        self.unlabeled_data = Subset(self.dataset, self.reverse_ids(labeled_ids))

    def __repr__(self):
        result = "[Active Pool]\n"
        result += f"Length of unlabeled set {len(self.unlabeled_data)} ({len(self.unlabeled_data)/len(self.dataset)*100:.1f}%).\n"
        result += f"Length of labeled set {len(self.labeled_data)} ({len(self.labeled_data)/len(self.dataset)*100:.1f}%)."
        return result

    def update(self, result: QueryResult):
        updated_ids = self.convert_to_original_ids(result.indices)
        self.labeled_data.indices  += updated_ids
        self.unlabeled_data.indices = self.reverse_ids(self.get_labeled_ids())

    def reset(self):
        self.unlabeled_data.indices = list(range(len(self.dataset)))
        self.labeled_data.indices   = []

    def get_unlabeled_dataset(self):
        return self.unlabeled_data

    def get_labeled_dataset(self):
        return self.labeled_data

    def get_unlabeled_ids(self):
        return self.unlabeled_data.indices

    def get_labeled_ids(self):
        return self.labeled_data.indices

    def get_unlabeled_dataloader(self, batch_size: Optional[int] = None, shuffle: bool = False):
        batch_size = batch_size if batch_size is not None else self.batch_size
        return DataLoader(self.unlabeled_data, batch_size, shuffle)

    def get_labeled_dataloader(self, batch_size: Optional[int] = None, shuffle: bool = True):
        batch_size = batch_size if batch_size is not None else self.batch_size
        return DataLoader(self.labeled_data, batch_size, shuffle)

    def reverse_ids(self, indices: List[int]):
        return list(set(range(len(self.dataset))) - set(indices))
    
    def convert_to_original_ids(self, indices: List[int]):
        return [self.unlabeled_data.indices[i] for i in indices]