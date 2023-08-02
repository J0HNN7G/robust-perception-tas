from __future__ import annotations
from typing import Any, List, Callable


class Analyzer():
    """
    Analyzer class for Autonomous Perception Robustness Testing Framework.

    Args:
        data (List[iter]): List of perception data to be analyzed.
        labels (List[iter]): List of perception labels corresponding to the data.
        augmentation (List[Callable]): List of augmentation functions to be applied to data.
        model (List[Callable]): List of models to be applied to data.
        metric (List[Callable]): List of metric functions to evaluate the analysis.

    Attributes:
        data (List[iter]): List of perception data to be analyzed.
        labels (List[iter]): List of perception labels corresponding to the data.
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
        data: List[iter], 
        labels: List[iter],
        augmentation: List[Callable],
        model: List[Callable],
        metric: List[Callable]
    ) -> None:
        """
        Initializes the Analyzer object.

        Args:
            data (List[iter]): List of perception data to be analyzed.
            labels (List[iter]): List of perception labels corresponding to the data.
            augmentation (List[Callable]): List of augmentation functions to be applied to data.
            model (List[Callable]): List of models to be applied to data.
            metric (List[Callable]): List of metric functions to evaluate the analysis.

        Raises:
            ValueError: If the length of any argument is not equal to the length of the data.
        """
        for arg in [labels, augmentation, model, metric]:
            if len(arg) != len(data):
                raise ValueError('all arguments should have the same length')

        self.data = data
        self.labels = labels
        self.augmentation = augmentation
        self.model = model
        self.metric = metric
        self.cache = None

    def cache(self, cache: Any) -> None:
        """
        Caches the given value.

        Args:
            cache (Any): Value to be cached.
        """
        self.cache = cache

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
            dict[str, Any]: A dictionary containing components for next step of analysis.
        """
        curr_idx = self.idx
        self.idx = (self.idx + 1) % len(self.data)

        result = {
            'data': self.data[curr_idx],
            'labels': self.labels[curr_idx],
            'augmentation': self.augmentation[curr_idx],
            'model': self.model[curr_idx],
            'metric': self.metric[curr_idx],
            'cache' : self.cache
        }
        return result

    def __len__(self) -> int:
        """
        Returns the number of analysis steps.

        Returns:
            int: Returns the number of analysis steps.
        """
        return len(self.data)