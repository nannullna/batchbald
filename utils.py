from typing import Any, List, Optional, Dict
from dataclasses import dataclass, field
import random

import numpy as np
import torch

@dataclass
class QueryResult:
    """A returned type of the class `ActiveQuery`"""
    scores: List[float]
    indices: List[int]
    labels: List[int] = field(init=False)
    info: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self):
        return f"[QueryResult] length {len(self.indices)}."

    def is_labeled(self):
        return bool(self.labels)

    def set_labels(self, labels):
        if len(self.scores) != len(labels):
            raise ValueError("The length of the query and that of the labels provided do not match.")
        self.labels = list(labels)

def set_all_seeds(seed, verbose=False):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

    if verbose:
        print("All random seeds set to", seed)