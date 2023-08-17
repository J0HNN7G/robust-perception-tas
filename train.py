# code altered from TorchVision 0.3 Object Detection Finetuning Tutorial
# http://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
# code altered from MIT semantic segmentation repository
# https://github.com/CSAILVision/semantic-segmentation-pytorch

# system libs
import os
import sys
import argparse

# logging
import logging
import traceback
from contextlib import redirect_stdout

# object detection
import torch
from torchvision.transforms import ToPILImage

# seeding randomness
import random
import numpy as np

# training
from aprtf.config import cfg
from aprtf.models import ModelBuilder, OptimizerBuilder, LRScheduleBuilder
from aprtf.dataset import PedestrianDetectionDataset,  get_transform
from aprtf.references.engine import train_one_epoch, evaluate
from aprtf.references.utils import collate_fn
from aprtf.visuals import visualize_results, FIG_NUM_IMAGES


# constants
TRAIN_NAME = 'train'
VAL_NAME = 'val'
TRAIN_EPOCH_NAME = f'{TRAIN_NAME}/epoch'
TRAIN_LR_NAME = f'{TRAIN_NAME}/lr_last_iter'
VAL_AP_NAME = f'{VAL_NAME}/mAP_0.5:0.95:0.05'
VAL_AR_NAME = f'{VAL_NAME}/mAR_0.5:0.95:0.05'

BEST_EPOCH_NAME = 'best_' + TRAIN_EPOCH_NAME
BEST_AP_NAME = 'best_' + VAL_AP_NAME

WEIGHT_FN = 'weights_epoch_{:d}.pth'
WEIGHT_FINAL_FN = 'weights_final.pth'
WEIGHT_BEST_FN = 'weights_best.pth'

TRAIN_HEADERS = [TRAIN_EPOCH_NAME, TRAIN_LR_NAME]
VAL_HEADERS = [VAL_AP_NAME, VAL_AR_NAME]
SEP = '\t'


def setup_previous_history(history, cfg):
    # catch up on history
    with open(cfg.TRAIN.history, 'r') as f:
        lines = f.readlines()
        # update headers
        cfg.TRAIN.history_headers = lines[0].split(SEP)
        # keep track of best epoch
        val_ap_idx = cfg.TRAIN.history_headers.index(VAL_AP_NAME)
        for row in f.readlines()[1:]:
            vals = row.split(SEP)
            row_val_ap = float(vals[val_ap_idx])
            if history[BEST_AP_NAME] < row_val_ap:
                epoch_idx = cfg.TRAIN.history_headers.index(TRAIN_EPOCH_NAME)
                history[BEST_EPOCH_NAME] = int(epoch_idx)
                history[BEST_AP_NAME] = row_val_ap


def setup_loss_details(train_log, cfg):
    loss_names = [loss_name for loss_name in train_log.meters.keys() if 'loss' in loss_name]
    loss_headers = [TRAIN_NAME + '/' + loss_name for loss_name in loss_names]
    # check same loss headers if part of partial run
    if cfg.TRAIN.LEN.start_epoch > 0:
        loss_headers_prev = cfg.TRAIN.history_headers[len(TRAIN_HEADERS):-len(VAL_HEADERS)]
        for i in range(len(cfg.TRAIN.history_headers)):
            assert loss_headers_prev[i] == loss_headers[i], 'Loss headers have changed!'
    else:
        cfg.TRAIN.history_headers = TRAIN_HEADERS + loss_headers + VAL_HEADERS
        with open(cfg.TRAIN.history, 'w') as f:
            f.write(SEP.join(cfg.TRAIN.history_headers) + '\n')
    return loss_names, loss_headers


def visual_evaluate(model, data_loader, cfg, device):
    logging.info('Saving visual')

    cpu_device = torch.device("cpu")
    model.eval()
    
    plot_images = []
    gt_bbs = []
    dt_bbs = [] 
    for images, targets in data_loader:
        # probably wrong way
        plot_image = ToPILImage()(images[0])
        plot_images.append(plot_image)

        gt_targets = [{k: v for k, v in t.items()} for t in targets]
        gt_bbs.append(gt_targets[0]['boxes'])

        input_images = [images[0].to(device)]
        with torch.no_grad():
            dt_targets = model(input_images)
            dt_targets = [{k: v.to(cpu_device) for k, v in t.items()} for t in dt_targets]
            dt_bbs.append(dt_targets[0]['boxes'])
        del input_images

        if len(plot_images) == FIG_NUM_IMAGES:
            break

    visualize_results(cfg.TRAIN.visual, plot_images, gt_bbs, dt_bbs)


def checkpoint(model, history, cfg, epoch):
    logging.info('Saving checkpoint')

    dict_model = model.state_dict()

    # save current model
    if epoch == (cfg.TRAIN.LEN.num_epoch-1):
        weights_fp = os.path.join(cfg.TRAIN.path, WEIGHT_FINAL_FN)
    else:
        weights_fp = os.path.join(cfg.TRAIN.path, WEIGHT_FN.format(epoch))
    torch.save(dict_model, weights_fp)

    # update best model
    if history[BEST_EPOCH_NAME] == epoch:
        torch.save(dict_model, os.path.join(cfg.TRAIN.path, WEIGHT_BEST_FN))

    # update history
    with open(cfg.TRAIN.history, 'a') as f:
        stats = SEP.join([str(history[x]) for x in cfg.TRAIN.history_headers]) + '\n'
        f.write(stats)

    # delete weights from the previous epoch
    if epoch > 0:
        prev_epoch = epoch - 1
        prev_weight_fn = os.path.join(cfg.TRAIN.path, WEIGHT_FN.format(prev_epoch))
        if os.path.exists(prev_weight_fn):
            os.remove(prev_weight_fn)
            logging.info(f'Previous weights (epoch {prev_epoch}) deleted')


def main(cfg, device):
    # model
    model = ModelBuilder.build_detector(args=cfg.MODEL,
                                        num_classes=cfg.DATASET.num_classes,
                                        weights=cfg.TRAIN.weights)
    model.to(device)

    # dataset
    train_path = os.path.join(cfg.DATASET.path,cfg.DATASET.LIST.train)
    train_dataset = PedestrianDetectionDataset(train_path, transforms=get_transform(True))
    val_path = os.path.join(cfg.DATASET.path,cfg.DATASET.LIST.val)
    val_dataset = PedestrianDetectionDataset(val_path, transforms=get_transform(False))

    # dataloaders
    train_data_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=cfg.TRAIN.DATA.batch_size, shuffle=True, num_workers=cfg.TRAIN.DATA.num_workers,
        collate_fn=collate_fn)
    val_data_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=cfg.TRAIN.DATA.batch_size, shuffle=False, num_workers=cfg.TRAIN.DATA.num_workers,
        collate_fn=collate_fn)

    # optimizer
    optimizer = OptimizerBuilder.build_optimizer(cfg.TRAIN.OPTIM, model)
    lr_scheduler = LRScheduleBuilder.build_scheduler(cfg.TRAIN.LR, optimizer)
    
    # track metrics
    history = {
        BEST_EPOCH_NAME: 0,
        BEST_AP_NAME: -1
    }
    # catch up
    if cfg.TRAIN.LEN.start_epoch > 0:
        # catch up on learning rate
        for i in range(0, cfg.TRAIN.LEN.start_epoch):
            lr_scheduler.step()
        # catch up on history
        setup_previous_history(history, cfg)

    # training
    for epoch in range(cfg.TRAIN.LEN.start_epoch, cfg.TRAIN.LEN.num_epoch):
        # early stopping
        if cfg.TRAIN.LEN.early_stop < epoch - history[BEST_EPOCH_NAME]:
            logging.info(f'Early stop! No improvement in validation set for {cfg.TRAIN.LEN.early_stop} epochs')
            # rename checkpoint to final weights
            curr_weights_fp = os.path.join(cfg.TRAIN.path, WEIGHT_FN.format(epoch-1))
            final_weights_fp = os.path.join(cfg.TRAIN.path, WEIGHT_FINAL_FN)
            os.rename(curr_weights_fp, final_weights_fp)
            break
        else:
            logging.info(f'Starting Epoch {epoch}')
            
        # train + evaluate
        with redirect_stdout(logging):
            train_log = train_one_epoch(model, optimizer, train_data_loader, device, epoch, print_freq=cfg.TRAIN.DATA.disp_iter)
            eval_log = evaluate(model, val_data_loader, device=device)
        
        # make loss headers in first epoch of run 
        if epoch == cfg.TRAIN.LEN.start_epoch:
            loss_names, loss_headers = setup_loss_details(train_log, cfg)

        # update history
        history[TRAIN_EPOCH_NAME] = epoch
        history[TRAIN_LR_NAME] = lr_scheduler.get_last_lr()[-1]

        # add training metrics
        for i in range(len(loss_names)):
            loss_name = loss_names[i]
            loss_header = loss_headers[i]
            loss = train_log.meters[loss_name].total
            history[loss_header] = loss 

        # add validation metrics
        history[VAL_AP_NAME] = eval_log.coco_eval['bbox'].stats[0]
        history[VAL_AR_NAME] = eval_log.coco_eval['bbox'].stats[6]
        if history[BEST_AP_NAME] < history[VAL_AP_NAME]:
            history[BEST_EPOCH_NAME] = epoch
            history[BEST_AP_NAME] = history[VAL_AP_NAME]

        # save model + history + visual
        checkpoint(model, history, cfg, epoch)
        visual_evaluate(model, val_data_loader, cfg, device=device)

        # update learning rate
        lr_scheduler.step()

    logging.info('Training Done!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="PyTorch Pedestrian Detection Finetuning"
    )
    parser.add_argument(
        "-c",
        required=True,
        metavar="FILENAME",
        help="absolute path to config file",
        type=str,
    )
    parser.add_argument(
        "-i",
        required=True,
        metavar="PATH",
        help="absolute path to directory with train and validation lists",
        type=str,
    )
    parser.add_argument(
        "-o",
        required=True,
        metavar="PATH",
        help="absolute path to checkpoint directory",
        type=str,
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()
    cfg.merge_from_file(args.c)
    cfg.merge_from_list(args.opts)
    cfg.DATASET.path = args.i
    cfg.TRAIN.path = args.o

    # check if already done
    final_weight_fp = os.path.join(cfg.TRAIN.path, WEIGHT_FINAL_FN)
    if os.path.exists(final_weight_fp):
        print(f'Training was done already! Final weights: {final_weight_fp}')
        exit()

    # make output directory
    if not os.path.isdir(cfg.TRAIN.path):
        os.makedirs(cfg.TRAIN.path)
    elif cfg.TRAIN.LEN.start_epoch == 0:
        # starting from scratch
        for f in os.listdir(cfg.TRAIN.path):
            os.remove(os.path.join(cfg.TRAIN.path,f))

    # set/save random seed
    if len(cfg.TRAIN.seed) == 0: 
        seed = torch.seed()
        cfg.TRAIN.seed = int(seed)
    else:
        cfg.TRAIN.SEED = int(cfg.TRAIN.seed)
        torch.manual_seed(cfg.TRAIN.seed)
    random.seed(cfg.TRAIN.seed) 
    # seed must be between 0 and 2**32 -1
    np.random.seed(cfg.TRAIN.seed % (2**32 - 1))

    # make config 
    config_fp = os.path.join(cfg.TRAIN.path, cfg.TRAIN.FN.cfg)
    if not os.path.exists(config_fp):
        with open(config_fp, 'w') as f:
            f.write(str(cfg))

    # setup logger
    log_fp = os.path.join(cfg.TRAIN.path, cfg.TRAIN.FN.log)
    if not os.path.exists(log_fp):
        open(log_fp, 'a').close()
    logging.basicConfig(level=logging.INFO,
                        format='[%(asctime)s %(levelname)s %(filename)s] %(message)s',
                        datefmt='%H:%M:%S',
                        handlers=[logging.FileHandler(log_fp)])
    # for redirecting stdout
    logging.write = lambda msg: logging.info(msg) if msg != '\n' else None
    # log details
    logging.info("Loaded configuration file: {}".format(args.c))
    logging.info("Running with config:\n{}".format(cfg))
    logging.info("Outputting to: {}".format(cfg.TRAIN.path))

    # start from checkpoint
    cfg.TRAIN.history = os.path.join(cfg.TRAIN.path, cfg.TRAIN.FN.hist)
    cfg.TRAIN.weights = ""
    if cfg.TRAIN.LEN.start_epoch > 0:
        cfg.TRAIN.weights = os.path.join(cfg.TRAIN.path, WEIGHT_FN.format(cfg.TRAIN.LEN.start_epoch-1))
        assert os.path.exists(cfg.TRAIN.weights), "weight checkpoint does not exist!"
        assert os.path.exists(cfg.TRAIN.history), "history checkpoint does not exist!"

    # make visual filepath
    cfg.TRAIN.visual = os.path.join(cfg.TRAIN.path, cfg.TRAIN.FN.vis)

    # train on the GPU or on the CPU, if a GPU is not available
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        logging.info('No GPU found! Training on CPU')
        device = torch.device('cpu')

    try:
        main(cfg, device)
    except Exception:
        logging.error(traceback.format_exc())
        # document everything
        with open(log_fp, 'r') as f:
            print(f.read())
        # for bash script
        sys.exit(1)
    sys.exit(0)


