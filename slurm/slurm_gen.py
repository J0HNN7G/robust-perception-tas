import argparse
import os

# Constants
USER_HOME = os.path.expanduser("~")
USER_NAME = os.environ["USER"]
SCRATCH_DISK = "/disk/scratch"
SCRATCH_HOME = os.path.join(SCRATCH_DISK, USER_NAME)
PROJECT = "git/robust-perception-tas"
HOME_PROJECT = os.path.join(USER_HOME, PROJECT)
SCRATCH_PROJECT = os.path.join(SCRATCH_HOME, PROJECT)

SLURM_DIR = "slurm"
CONFIG_DIR = "config"
INPUT_DIR = "data/sets"
OUTPUT_DIR = "ckpt"

GEN_FN = 'experiment.py'
TRAIN_FN = 'train.py'

def generate_experiments(config_fn, dataset_n, checkpoint_n):
    command = (
        f"python {os.path.join(HOME_PROJECT, SLURM_DIR, GEN_FN)}"
        f" -t {os.path.join(HOME_PROJECT, TRAIN_FN)}"
        f" -c {os.path.join(HOME_PROJECT, CONFIG_DIR, config_fn)}"
        f" -i {os.path.join(SCRATCH_PROJECT, INPUT_DIR, dataset_n)}"
        f" -o {os.path.join(SCRATCH_PROJECT, OUTPUT_DIR, checkpoint_n)}"
        f" -s {os.path.join(HOME_PROJECT, SLURM_DIR)}"
    )

    os.system(command)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=True, help="Config filename")
    parser.add_argument("-d", "--dataset", required=True, help="Dataset name")
    parser.add_argument("-n", "--checkpoint", required=True, help="Checkpoint name")
    args = parser.parse_args()

    config_fn = args.config
    dataset_n = args.dataset
    checkpoint_n = args.checkpoint

    generate_experiments(config_fn, dataset_n, checkpoint_n)

if __name__ == "__main__":
    main()