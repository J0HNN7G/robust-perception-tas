import os
import argparse

# constants
USER = 's1915791'
USER_HOME = os.path.join('home', USER)
PROJECT = 'git/robust-perception-tas'
PROJECT_PATH = os.path.join(USER_HOME, PROJECT)
SLURM_PATH = os.path.join(PROJECT_PATH, 'slurm') 
INPUT_PATH = os.path.join(PROJECT_PATH, 'data/sets')
OUTPUT_TXT = os.path.join(SLURM_PATH, "experiment.txt")
OUTPUT_TSV = os.path.join(SLURM_PATH, 'experiment.tsv')

FAILED_TSV_NAME = 'failed_experiments.tsv'
TIMEOUT_TSV_NAME = 'timeout_experiments.tsv'

TRANSFER_TO_PREFIX = "Moving input data to the compute node's scratch space: "
RUNNING_PREFIX = 'Running provided command: '
FAILED_PROMPT = 'Command failed!'
TRANSFER_FROM_PROMPT = "Moving output data back to DFS"
FINISHED_PROMPT = 'Job finished successfully!'
TIMEOUT_PROMPT = 'CANCELLED'


def main(args):
    # different job status
    queuing_ids = []
    transferTo_ids = []
    running_ids = []
    transferFrom_ids = []
    finished_ids = []
    failed_ids = []
    timeout_ids = []

    # get experiments
    lines = []
    with open(args.exp_txt, 'r') as f:
        lines = f.readlines()

    # look at status of each experiment
    for id, line in enumerate(lines, start=1):
        slurm_log_fp = os.path.join(args.log, f'slurm-{args.job}_{id}.out') 
        if not os.path.exists(slurm_log_fp):
            queuing_ids.append(id)
            continue

        process_flags = [False for _ in range(6)]
        with open(slurm_log_fp, 'r') as f:
            log_line = f.readline()
            while log_line:
                if TRANSFER_TO_PREFIX in log_line:
                    process_flags[0] = True
                elif (RUNNING_PREFIX + line) in log_line:
                    process_flags[1] = True
                elif TRANSFER_FROM_PROMPT  in log_line:
                    process_flags[2] = True
                elif FINISHED_PROMPT in log_line:
                    process_flags[3] = True
                    break
                elif FAILED_PROMPT in log_line:
                    process_flags[4] = True
                    break
                elif TIMEOUT_PROMPT in log_line:
                    process_flags[5] = True
                    break
                log_line = f.readline()

        # sort flags
        if process_flags[5]:
            timeout_ids.append(id)
        elif process_flags[4]:
            failed_ids.append(id)
        elif process_flags[3]:
            finished_ids.append(id)
        elif process_flags[2]:
            transferFrom_ids.append(id)
        elif process_flags[1]:
            running_ids.append(id)
        elif process_flags[0]:
            transferTo_ids.append(id)

    print('QUEUING ----------')
    print(queuing_ids)
    print()

    print('TRANSFER TO GPU NODE ----------')
    print(transferTo_ids)
    print()

    print('RUNNING ----------')
    print(running_ids)
    print()

    print('TRANSFER FROM GPU_NODE ----------')
    print(transferFrom_ids)
    print()

    print('FINISHED ----------')
    print(finished_ids)
    print()
    
    print('FAILED ----------')
    print(failed_ids)
    print()

    print('CANCELLED ----------')
    print(timeout_ids)
    print()

    any_fails = len(failed_ids) > 0
    any_timeouts = len(timeout_ids) > 0

    if any_fails or any_timeouts:
        # saving details
        lines = []
        with open(args.exp_tsv, 'r') as f:
            lines = f.readlines()

        if any_fails:
            failed_tsv = os.path.join(os.path.dirname(args.exp_tsv), FAILED_TSV_NAME)
            with open(failed_tsv, 'w') as f:
                # header
                f.write(lines[0])
                for id in failed_ids:
                    f.write(lines[id])
            print(f'Saved failed experiment details in: {failed_tsv}')

        if any_timeouts:
            timeout_tsv = os.path.join(os.path.dirname(args.exp_tsv), TIMEOUT_TSV_NAME)
            with open(timeout_tsv, 'w') as f:
                # header
                f.write(lines[0])
                for id in timeout_ids:
                    f.write(lines[id])
            print(f'Saved cancelled experiment details in: {timeout_tsv}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Slurm Experiment Checker"
    )
    parser.add_argument(
        "--job",
        required=True,
        metavar="INT",
        help="Slurm Job ID",
        type=int,
    )
    parser.add_argument(
        "--log",
        default=SLURM_PATH,
        metavar="PATH",
        help="Slurm Job ID",
        type=str,
    )
    parser.add_argument(
        "--exp_txt",
        default=OUTPUT_TXT,
        metavar="PATH",
        help="Experiments text file",
        type=str,
    )
    parser.add_argument(
        "--exp_tsv",
        default=OUTPUT_TSV,
        metavar="PATH",
        help="Experiments TSV",
        type=str,
    )
    args = parser.parse_args()
    if args.job < 1:
        raise ValueError('Job ID is not positive!')
    if not os.path.exists(args.log):
        raise FileNotFoundError('Slurm logs not found!') 
    if not os.path.exists(args.exp_txt):
        raise FileNotFoundError('Experiments text file not found!') 
    if not os.path.exists(args.exp_tsv):
        raise FileNotFoundError('Experiments TSV file not found!') 
    
    main(args)