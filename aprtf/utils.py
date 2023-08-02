# code from MIT semantic segmentation repository
# https://github.com/CSAILVision/semantic-segmentation-pytorch

# system libs
import sys
import logging

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