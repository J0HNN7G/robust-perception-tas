#!/usr/bin/env python3
"""Script for generating experiments.txt"""
import os
import json
import argparse

# grid generation
import numpy as np

# constants
LOC_OPTS = ['PERSONAL', 'EDI']

PARAM_LIST = ['DATA.batch_size', 'OPTIM.lr', 'OPTIM.momentum', 'OPTIM.weight_decay', 'LR.step_size', 'LR.gamma']
PARAM_CALLS = ['TRAIN.' + x for x in PARAM_LIST]
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
        "-c", "--config",
        required=True,
        metavar="PATH",
        help="Absolute path to path config file",
        type=str,
    )
    parser.add_argument(
        "-l", "--loc",
        required=True,
        metavar="STR",
        choices=LOC_OPTS,
        help="Working directory [PERSONAL,EDI]",
        type=str,
    )

    args = parser.parse_args()
    if not os.path.exists(args.config):
        raise FileNotFoundError('Config file not found!')
    cfg = {}
    with open(args.config, 'r') as f:
        cfg = json.load(f)

    INPUT_HOME = cfg[args.loc]['HOME']
    INPUT_USER = cfg[args.loc]['USER']
    INPUT_PROJECT = cfg[args.loc]['PROJECT']

    # output on scratch disk
    if args.loc == LOC_OPTS[1]:
        OUTPUT_HOME = cfg['SCRATCH']['HOME']
        OUTPUT_USER = cfg['SCRATCH']['USER']
        OUTPUT_PROJECT = cfg['SCRATCH']['PROJECT']
    else:
        OUTPUT_HOME = INPUT_HOME
        OUTPUT_USER = INPUT_USER
        OUTPUT_PROJECT = INPUT_PROJECT
        
    exp_name = f"{cfg['ARCH']}-{cfg['DATASET'].lower()}" 
    input_project_path = os.path.join(INPUT_HOME, INPUT_USER, INPUT_PROJECT)
    input_train_path = os.path.join(input_project_path, cfg['TRAIN_FN'])
    input_config_path = os.path.join(input_project_path, cfg['CONFIG_DN'], f"{exp_name}.yaml" )
    input_data_path = os.path.join(input_project_path, cfg['DATA_DN'], cfg['DATASET'])
    output_ckpt_path = os.path.join(OUTPUT_HOME, OUTPUT_USER, OUTPUT_PROJECT, cfg['CKPT_DN'], exp_name)

    base_call = f"python {input_train_path} -c {input_config_path} -i {input_data_path} -o {output_ckpt_path}"

    # parameters
    batch_size = [1, 128]
    lrs = [0.005]
    step_sizes = [3]
    gammas = [0.1]
    momentums = [0.9]
    weight_decays = [0.0005] 

    param_list = [batch_size, lrs, momentums, weight_decays, step_sizes, gammas]
    settings = cartesian(param_list)

    # generation
    nr_expts = len(settings)
    print(f'Total experiments = {nr_expts}')
    print(f'Estimated time = {(nr_expts / NR_SERVERS * AVG_EXPT_TIME)/60} hrs')


    input_slurm_path = os.path.join(input_project_path, cfg['SLURM_DN'])
    input_exp_txt_path = os.path.join(input_slurm_path, cfg['EXP']['TXT_FN'])
    input_exp_tsv_path = os.path.join(input_slurm_path, cfg['EXP']['TSV']['DEFAULT_FN'])
    # clear tsv and create header
    with open(input_exp_tsv_path, 'w') as f:
        header =  SEP.join([IDX_NAME] + PARAM_LIST) + '\n'
        f.write(header)
    # create/clear experiments file
    open(input_exp_txt_path, 'w').close()

    for i, params in enumerate(settings, start=1):
        param_call_str = ' '.join(f"{param_call} {param}" for param_call, param in zip(PARAM_CALLS, params))
        # Note that we don't set a seed for rep - a seed is selected at random
        # and recorded in the output data by the python script
        expt_call = f"{base_call}_{i} {param_call_str}\n"
        
        with open(input_exp_txt_path, 'a') as f:
            f.write(expt_call)
        with open(input_exp_tsv_path, 'a') as f:
            line = SEP.join([str(i)] + [str(x) for x in params]) + '\n'
            f.write(line)