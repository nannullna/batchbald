from typing import Optional, Union, List, Dict, NoReturn
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset
from utils import QueryResult

class ActivePool:

    dataset: Dataset
    batch_size: int
    labeled_data: Subset
    unlabeled_data: Subset

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int = 1,
        labeled_ids: Optional[List[int]] = None,
        query_set: Optional[Dataset] = None,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.query_set = query_set or self.dataset
        
        labeled_ids = list(labeled_ids) if labeled_ids else []

        self.labeled_data   = Subset(self.dataset, labeled_ids)
        self.unlabeled_data = Subset(self.query_set, self.reverse_ids(labeled_ids))
        # query_set is expected not to apply augmentations for better uncertainty estimations and query results

    def __repr__(self):
        result = "[Active Pool]\n"
        result += f"Length of unlabeled set {len(self.unlabeled_data)} ({len(self.unlabeled_data)/len(self.dataset)*100:.1f}%).\n"
        result += f"Length of labeled set {len(self.labeled_data)} ({len(self.labeled_data)/len(self.dataset)*100:.1f}%)."
        return result

    def update(self, result: QueryResult):
        updated_ids = self.convert_to_original_ids(result.indices)
        self.labeled_data.indices  += updated_ids
        self.unlabeled_data.indices = self.reverse_ids(self.get_labeled_ids())

    def reset(self) -> NoReturn:
        self.unlabeled_data.indices = list(range(len(self.dataset)))
        self.labeled_data.indices   = []

    def get_unlabeled_dataset(self) -> Subset:
        return self.unlabeled_data

    def get_labeled_dataset(self) -> Subset:
        return self.labeled_data

    def get_unlabeled_ids(self) -> List[int]:
        return self.unlabeled_data.indices

    def get_labeled_ids(self) -> List[int]:
        return self.labeled_data.indices

    def get_unlabeled_targets(self) -> Union[List[int], None]:
        if hasattr(self.dataset, "targets"):
            # torchvision.datasets (MNIST, CIFAR10, CIFAR100, ...)
            all_targets = np.asarray(self.dataset.targets)
            return all_targets[self.get_unlabeled_ids()].tolist()

    def get_labeled_targets(self) -> Union[List[int], None]:
        if hasattr(self.dataset, "targets"):
            # torchvision.datasets (MNIST, CIFAR10, CIFAR100, ...)
            all_targets = np.asarray(self.dataset.targets)
            return all_targets[self.get_labeled_ids()].tolist()

    def get_unlabeled_dataloader(self, batch_size: Optional[int] = None, shuffle: bool = False):
        batch_size = batch_size if batch_size is not None else self.batch_size
        return DataLoader(self.unlabeled_data, batch_size, shuffle, num_workers=1, pin_memory=True)

    def get_labeled_dataloader(self, batch_size: Optional[int] = None, shuffle: bool = True):
        batch_size = batch_size if batch_size is not None else self.batch_size
        return DataLoader(self.labeled_data, batch_size, shuffle, num_workers=1, pin_memory=True)

    def reverse_ids(self, indices: List[int]):
        return list(set(range(len(self.dataset))) - set(indices))
    
    def convert_to_original_ids(self, indices: List[int]):
        return [self.unlabeled_data.indices[i] for i in indices]