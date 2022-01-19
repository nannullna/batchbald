from typing import List, Optional, Dict
from dataclasses import dataclass, field

import numpy as np

import torch

@dataclass
class QueryResult:
    """A returned type of the class `ActiveQuery`"""
    scores: List[float]
    indices: List[int]
    labels: List[int] = field(init=False)
    info: Optional[Dict] = None

    def __repr__(self):
        return f"[QueryResult] length {len(self.indices)}."

    def is_labeled(self):
        return bool(self.labels)

    def set_labels(self, labels):
        if len(self.scores) != len(labels):
            raise ValueError("The length of the query and that of the labels provided do not match.")
        self.labels = list(labels)

