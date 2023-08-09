
# args
import os
import argparse

from aprtf.config import cfg

# help
import logging
from train import TRAIN_NAME, VAL_AP_NAME, SEP
import PIL as Image

# wandb, api key should be give prior to script 
import wandb

# Constants
BEST_VAL_NAME = VAL_AP_NAME + '_best'


def main(cfg):
    run = wandb.init(
        project='robust-perception-tas',
        name=cfg.DIR.split('/')[-1],
        config = {
            'architecture' : cfg.MODEL.arch,
            'dataset' : cfg.DATASET.root_dataset.split('/')[-1],
            f'{TRAIN_NAME}/batch_size' : cfg.TRAIN.batch_size, 
            f'{TRAIN_NAME}/epochs' : cfg.TRAIN.num_epoch,
            f'{TRAIN_NAME}/optimizer' : cfg.TRAIN.optim,
            f'{TRAIN_NAME}/optimizer_momentum' : cfg.TRAIN.momentum, 
            f'{TRAIN_NAME}/optimizer_weight_decay' : cfg.TRAIN.weight_decay, 
            f'{TRAIN_NAME}/initial_learning_rate': cfg.TRAIN.lr,
            f'{TRAIN_NAME}/learning_rate_step_size' : cfg.TRAIN.lr_step_size,
            f'{TRAIN_NAME}/learning_rate_gamma' : cfg.TRAIN.lr_gamma,
            f'{TRAIN_NAME}/early_stopping_threshold_epochs' : cfg.TRAIN.early_stop,
            f'{TRAIN_NAME}/seed' : cfg.TRAIN.seed, 
        }
    )

    with open(cfg.MODEL.history, 'r') as f:
            lines = f.readlines()
            # ignore epoch
            headers = lines[0].split(SEP)[1:]

            best_val_ap = -1
            val_ap_idx = headers.index(VAL_AP_NAME)

            for content in lines[1:]:
                content = content.split(SEP)[1:]
                vals = [float(x) for x in content]
                # log row
                val_dict = dict(list(zip(headers,vals)))
                run.log(val_dict)
                # update summary
                if best_val_ap < vals[val_ap_idx]:
                    best_val_ap = vals[val_ap_idx]
    
    # log visual
    im_fp = os.path.join(cfg.DIR, cfg.VAL.visual_name)
    wandb.log({"val/examples": wandb.Image(im_fp)})

    run.summary.update({BEST_VAL_NAME : best_val_ap})
    run.finish()
    
    logging.info('Wandb Logging Done!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Wandb Logging for PyTorch Pedestrian Detection Finetuning"
    )
    parser.add_argument(
        "--ckpt",
        default="ckpt/fasterrcnn_resnet50_fpn-pennfudan",
        metavar="FILE",
        help="path to model checkpoint directory",
        type=str,
    )
    args = parser.parse_args()
    cfg_fp = os.path.join(args.ckpt, 'config.yaml')
    assert os.path.exists(cfg_fp), 'config.yaml does not exist!'
    cfg.merge_from_file(cfg_fp)

    # setup logger
    cfg.MODEL.log = os.path.join(cfg.DIR, cfg.MODEL.log_name)
    cfg.MODEL.history = os.path.join(cfg.DIR, cfg.MODEL.history_name)
    assert os.path.exists(cfg.MODEL.log), 'logs do not exist!'
    assert os.path.exists(cfg.MODEL.history), 'history does not exist!'
    logging.basicConfig(level=logging.INFO,
                        format='[%(asctime)s %(levelname)s %(filename)s] %(message)s',
                        datefmt='%H:%M:%S',
                        handlers=[logging.FileHandler(cfg.MODEL.log)])
    logging.info(f"Starting Wandb Logging for experiment {cfg.DIR.split('/')[-1]}")

    main(cfg)