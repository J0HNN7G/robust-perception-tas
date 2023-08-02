# code altered from TorchVision 0.3 Object Detection Finetuning Tutorial
# http://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
# code altered from MIT semantic segmentation repository
# https://github.com/CSAILVision/semantic-segmentation-pytorch

# - add history adopts from previous history
# - add load latest config (looks for latest epoch)
# - make slurm compatible (save config, and weights)
# - make wandb compatible (log training and testing)
# - replace with nuscenes dataset (make the switch)

# system libs
import os
import random
import argparse

# object detection
import numpy as np
import torch

# training
from aprtf.models import ModelBuilder
from aprtf.dataset import PedestrianDetectionDataset, get_transform
from aprtf.references.engine import train_one_epoch, evaluate
from aprtf.references.utils import collate_fn

# config and log
from aprtf.config import cfg
from aprtf.utils import setup_logger


def checkpoint(model, train_log, coco_eval, cfg, epoch):
    print('Saving checkpoints...')

    dict_model = model.state_dict()

    #torch.save(
    #    history,
    #    f'{cfg.DIR}/history_epoch_{epoch}.pth')
    torch.save(
        dict_model,
        f'{cfg.DIR}/weights_epoch_{epoch}.pth')

    # delete weights from the previous epoch
    if epoch > 0:
        prev_epoch = epoch - 1
        prev_weight_file = f'{cfg.DIR}/weights_epoch_{prev_epoch}.pth'
        if os.path.exists(prev_weight_file):
            os.remove(prev_weight_file)
            print(f'Previous weights (epoch {prev_epoch}) deleted')


def main(cfg, device):
    model = ModelBuilder.build_detector(arch=cfg.MODEL.arch,
                                        num_classes=cfg.DATASET.num_classes,
                                        weights=cfg.MODEL.weights)
    model.to(device)

    # Dataset and Loader
    dataset = PedestrianDetectionDataset(cfg.DATASET.root_dataset, get_transform(train=True))
    dataset_test = PedestrianDetectionDataset(cfg.DATASET.root_dataset, get_transform(train=False))

    # split the dataset in train and test set
    indices = torch.randperm(len(dataset)).tolist()
    dataset = torch.utils.data.Subset(dataset, indices[:-50])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=True, num_workers=cfg.TRAIN.num_workers,
        collate_fn=collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=cfg.TRAIN.num_workers,
        collate_fn=collate_fn)

    # optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=cfg.TRAIN.lr,
                                momentum=cfg.TRAIN.momentum, 
                                weight_decay=cfg.TRAIN.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=cfg.TRAIN.lr_step_size,
                                                   gamma=cfg.TRAIN.lr_gamma)
    
    history = {'epoch': [], 
               'train_avg_loss': [], 
               'val_avg_ap_iou': [],
               'best_checkpoint': 0}
    for epoch in range(cfg.TRAIN.start_epoch, cfg.TRAIN.num_epoch):
        train_log = train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        # update learning rate
        lr_scheduler.step()

        # coco evaluate on validation data
        eval_log = evaluate(model, data_loader_test, device=device)

        # history
        history['epoch'].append(epoch)
        history['train_avg_loss'].append(train_log.meters['loss'].global_avg())
        history['val_avg_ap_iou'].append(eval_log.coco_eval.stats[0])
        history['best_checkpoint'] = history['epoch'][np.argmax(history['val_avg_ap_iou'])]

        # saving stuff
        checkpoint(model, history, cfg, epoch)

    print('Training Done!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="PyTorch Pedestrian Detection Finetuning Package"
    )
    parser.add_argument(
        "--cfg",
        default="config/fasterrcnn_resnet50_fpn_v2-pennfudan.yaml",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()
    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)

    logger = setup_logger(distributed_rank=0)
    logger.info("Loaded configuration file {}".format(args.cfg))
    logger.info("Running with config:\n{}".format(cfg))

    # output directory
    if not os.path.isdir(cfg.DIR):
        os.makedirs(cfg.DIR)
    logger.info("Outputting checkpoints to: {}".format(cfg.DIR))
    with open(os.path.join(cfg.DIR, 'config.yaml'), 'w') as f:
        f.write("{}".format(cfg))

    # start from checkpoint
    if cfg.TRAIN.start_epoch > 0:
        cfg.MODEL.weights = os.path.join(cfg.DIR, f'weights_epoch_{cfg.TRAIN.start_epoch}.pth')
        assert os.path.exists(cfg.MODEL.weights), "checkpoint does not exist!"

    # random seed
    random.seed(cfg.TRAIN.seed)
    np.random.seed(cfg.TRAIN.seed)
    torch.manual_seed(cfg.TRAIN.seed)

    # train on the GPU or on the CPU, if a GPU is not available
    if torch.cuda.is_available():
        device = torch.device('cuda')
        torch.cuda.empty_cache()
    else:
        device = torch.device('cpu')

    main(cfg, device)
