from typing import Optional, List

from torch.utils.data import Dataset, DataLoader, Subset

from utils import CandidateBatch

class ActiveLearningDataset:
    def __init__(
        self, 
        dataset: Dataset, 
        batch_size: int=1, 
        unlabeled_ids: Optional[List[int]]=None, 
        labeled_ids: Optional[List[int]]=None
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.unlabeled_ids = unlabeled_ids
        self.labeled_ids = labeled_ids

        self.unlabeled_data = Subset(self.dataset, self.unlabeled_ids)
        self.labeled_data   = Subset(self.dataset, self.labeled_ids)

        self.reset()
        self._update_dataset()

    def _update_dataset(self):
        self.unlabeled_data.indices = self.unlabeled_ids
        self.labeled_data.indices   = self.labeled_ids

    def update(self, candidates: CandidateBatch):
        self.labeled_ids += candidates.indices
        self.unlabeled_ids = list(set(range(len(self.dataset))) - set(self.labeled_ids))
        self._update_dataset()

    def reset(self):
        self.unlabeled_ids = list(range(len(self.dataset)))
        self.labeled_ids = []
        self._update_dataset()

    def get_unlabeled_dataset(self):
        return self.unlabeled_data

    def get_labeled_dataset(self):
        return self.labeled_data

    def get_unlabeled_indices(self):
        return self.unlabeled_ids

    def get_labeled_indices(self):
        return self.labeled_ids

    def get_unlabeled_dataloader(self, batch_size: Optional[int]=None, shuffle: bool=False):
        batch_size = batch_size if batch_size is not None else self.batch_size
        return DataLoader(self.unlabeled_data, batch_size, shuffle)

    def get_labeled_dataloader(self, batch_size: Optional[int]=None, shuffle: bool=True):
        batch_size = batch_size if batch_size is not None else self.batch_size
        return DataLoader(self.labeled_data, batch_size, shuffle)