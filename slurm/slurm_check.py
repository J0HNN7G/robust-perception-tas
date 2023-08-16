
import os
import argparse

from slurm_gen import PROJECT

USER = 's1915791'
SLURM_LOG_DIR = os.path.join(USER, 'home', 'slurm_logs')
CKPT_DIR = os.path.join(PROJECT, 'ckpt')

RUNNING_PROMPT = 'Running provided command'
ENDING_PROMPT = 'Job finished'

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Slurm Experiment Checker"
    )
    parser.add_argument(
        "--arch",
        default="fasterrcnn_resnet50_fpn",
        metavar="STR",
        help="Object detection architecture",
        type=str,
    )
    parser.add_argument(
        "--dataset",
        default="pennfudanped",
        metavar="DIRECTORY",
        help="path to intended dataset directory",
        type=str,
    )
    args = parser.parse_args()
