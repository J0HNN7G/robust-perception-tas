"""Analyzer class for multi-step analysis"""
# typing
from __future__ import annotations
from typing import Any, List, Callable

# structure
import torch

from dataset import CacheDataset

# logging
from tqdm import tqdm
from references.utils import SmoothedValue


class Analyzer():
    """
    Analyzer class for Autonomous Perception Robustness Testing Framework.

    Attributes:
        data (List[torch.utils.data.Dataset]): List of perception data to be analyzed.
        augmentation (List[Callable]): List of augmentation functions to be applied to data.
        model (List[Callable]): List of models to be applied to data.
        metric (List[Callable]): List of metric functions to evaluate the analysis.
        cache (Any): Caching variable for storing intermediate results.

    Raises:
        ValueError: If the length of any argument is not equal to the length of the data.

    Methods:
        cache: Caches the given value.
        __iter__: Initializes the iterator.
        __next__: Returns the next step of analysis.
        __len__: Returns the number of analysis steps.
    """
    def __init__(
        self, 
        data: List[torch.utils.data.Dataset],
        augmentation: List[Callable],
        model: List[Callable],
        metric: List[Callable]
    ) -> None:
        """
        Initializes the Analyzer object.

        Args:
            data (List[torch.utils.data.Dataset]): List of perception data to be analyzed.
            augmentation (List[Callable]): List of augmentation functions to be applied to data.
            model (List[Callable]): List of models to be applied to data.
            metric (List[Callable]): List of metric functions to evaluate the analysis.

        Raises:
            ValueError: If the length of any argument is not equal to the length of the data.
        """
        for arg in [augmentation, model, metric]:
            if len(arg) != len(data):
                raise ValueError('all arguments should have the same length')

        self.data = data
        self.augmentation = augmentation
        self.model = model
        self.metric = metric
        self.cache = [CacheDataset() for _ in range(len(data))]


    def __iter__(self) -> Analyzer:
        """
        Initializes the iterator.

        Returns:
            Analyzer: The Analyzer object itself.
        """
        self.idx = 0
        return self
    

    def __next__(self) -> dict[str, Any]:
        """
        Returns the next analysis result.

        Returns:
            SmoothedValue: An object containing results for the metric at next step of analysis.
        """
        curr_idx = self.idx
        self.idx = (self.idx + 1) % len(self.data)

        data = self.data[curr_idx]
        model = self.model[curr_idx]
        aug = self.augmentation[curr_idx]
        metric = self.metric[curr_idx]
        cache = self.cache[curr_idx]

        for sample, target in tqdm(data):
            pred = model(sample)
            if aug is not None:
                # robustness testing
                aug_pred = model(aug(sample))
            elif curr_idx > 0:
                # multi-step analysis
                self.cache[curr_idx-1] 
                aug_pred = model()
            else:
                # later-stage augmentation
                aug_pred = pred
            result = metric(pred, aug_pred, target)

            cache.update(aug_pred, target)
        
        return -1


    def __len__(self) -> int:
        """
        Returns the number of analysis steps.

        Returns:
            int: Returns the number of analysis steps.
        """
        return len(self.data)