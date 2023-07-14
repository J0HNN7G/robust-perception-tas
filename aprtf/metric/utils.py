import math
# #!/usr/bin/env python3
# This file is code from the MIT semantic segmentation repository
# https://github.com/CSAILVision/semantic-segmentation-pytorch

# system libs
import sys
import logging
# numerical libs
import math


def setup_logger(distributed_rank=0, filename="log.txt"):
    """
    Set up a logger to write log messages to a file and/or console.

    Args:
        distributed_rank (int): The rank of the current process in a distributed environment. If it is greater than
            zero, log messages will not be written to the console. Default is 0.
        filename (str): The name of the file to write log messages to. Default is "log.txt".

    Returns:
        logging.Logger: The logger object that can be used to write log messages.
    """
    logger = logging.getLogger("Logger")
    logger.setLevel(logging.DEBUG)
    # don't log results for the non-master process
    if distributed_rank > 0:
        return logger
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"
    ch.setFormatter(logging.Formatter(fmt))
    logger.addHandler(ch)

    return logger


class MetricMeter(object):
    """
    Computes and stores the average, current value, and standard deviation of a metric.

    Attributes:
        initialized (bool): True if the meter has been initialized, False otherwise.
        val (float or None): The current value of the metric.
        avg (float or None): The average value of the metric.
        sum (float or None): The sum of the metric values.
        count (float or None): The number of times the metric has been updated.
        std (float or None): The standard deviation of the metric.
    """

    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None
        self.std = None

    def initialize(self, val, weight):
        """
        Initialize the meter with an initial value and weight.

        Args:
            val (float): The initial value of the metric.
            weight (float): The weight of the initial value.

        Returns:
            None
        """
        self.val = val
        self.avg = val
        self.sum = val * weight
        self.count = weight
        self.std = 0
        self.initialized = True

    def update(self, val, weight=1):
        """
        Update the meter with a new value and weight.

        Args:
            val (float): The new value of the metric.
            weight (float): The weight of the new value. Default is 1.

        Returns:
            None
        """
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.add(val, weight)

    def add(self, val, weight):
        """
        Add a new value and weight to the meter.

        Args:
            val (float): The new value to add.
            weight (float): The weight of the new value.

        Returns:
            None
        """
        self.val = val
        self.sum += val * weight
        self.count += weight
        self.avg = self.sum / self.count
        self.std = math.sqrt(((self.count - 1) * self.std ** 2 + weight * (val - self.avg) ** 2) / self.count)

    def value(self):
        """
        Get the current value of the meter.

        Returns:
            float or None: The current value of the meter.
        """
        return self.val

    def average(self):
        """
        Get the average value of the meter.

        Returns:
            float or None: The average value of the meter.
        """
        return self.avg

    def standard_deviation(self):
        """
        Get the standard deviation of the meter.

        Returns:
            float or None: The standard deviation of the meter.
        """
        return self.std