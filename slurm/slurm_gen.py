#!/usr/bin/env python3
"""Script for generating experiments.txt"""
import os
import argparse

# grid generation
import numpy as np

# constants
PROJECT = 'robust-perception-tas'

SLURM_DIR = 'slurm'
INPUT_DIR = 'data/sets'
# This may need changing to e.g. /disk/scratch_fast depending on the cluster
SCRATCH_DISK = '/disk/scratch'  

PARAM_LIST = ['batch_size', 'lr', 'lr_step_size', 'lr_gamma', 'momentum', 'weight_decay']
PARAM_CALLS = ['TRAIN.' + x for x in PARAM_LIST]
OUTPUT_CALLS = os.path.join(SLURM_DIR, "experiment.txt")
OUTPUT_TSV = os.path.join(SLURM_DIR, 'experiment.tsv')
IDX_NAME = 'idx'
SEP = '\t'

NR_SERVERS = 10
# in minutes
AVG_EXPT_TIME = 40


def cartesian(arrays):
    """
    Generate a Cartesian product of input arrays.
    """
    arrays = [np.asarray(x) for x in arrays]
    dtype = [x.dtype for x in arrays]

    n = np.prod([x.size for x in arrays])
    product = []

    m = int(n / arrays[0].size)
    first_column = np.repeat(arrays[0], m)
    if len(arrays) > 1:
        rest_of_product = cartesian(arrays[1:])
        for j in range(arrays[0].size):
            product.extend([(first_column[j*m],) + t for t in rest_of_product])
    else:
        product.extend([(x,) for x in first_column])
    return product


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Slurm Grid Search Experiment Setup for APRTF"
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
    name_prefix = f'{args.arch}-{args.dataset}'
    base_call = f"python train.py --cfg config/{name_prefix}.yaml "

    USER = os.getenv('USER')
    # The home dir on the node's scratch disk
    SCRATCH_HOME = f'{SCRATCH_DISK}/{USER}'
    DATA_HOME = f'{SCRATCH_HOME}/{PROJECT}/{INPUT_DIR}'

    # parameters
    batch_size = (2 ** np.arange(0,6))[::-1]
    lrs = [0.005]
    lr_step_sizes = [3]
    gammas = [0.1]
    momentums = [0.9]
    weight_decays = [0.0005] 

    param_list = [batch_size, lrs, lr_step_sizes, gammas, momentums, weight_decays]
    settings = cartesian(param_list)

    # generation
    nr_expts = len(settings)
    print(f'Total experiments = {nr_expts}')
    print(f'Estimated time = {(nr_expts / NR_SERVERS * AVG_EXPT_TIME)/60} hrs')

    # clear tsv and create header
    with open(OUTPUT_TSV, 'w') as f:
        header =  SEP.join([IDX_NAME] + PARAM_LIST) + '\n'
        f.write(header)
    # create/clear experiments file
    open(OUTPUT_CALLS, 'w').close()

    for i, params in enumerate(settings, start=1):
        param_call_str = ' '.join(f"{param_call} {param}" for param_call, param in zip(PARAM_CALLS, params))
        # Note that we don't set a seed for rep - a seed is selected at random
        # and recorded in the output data by the python script
        expt_call = f"{base_call}{param_call_str} DIR ckpt/{name_prefix}_{i}\n"
        
        with open(OUTPUT_CALLS, 'a') as f:
            f.write(expt_call)
        with open(OUTPUT_TSV, 'a') as f:
            line = SEP.join([str(i)] + [str(x) for x in params]) + '\n'
            f.write(line)