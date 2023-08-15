# code altered from TorchVision 0.3 Object Detection Finetuning Tutorial
# http://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
# code altered from MIT semantic segmentation repository
# https://github.com/CSAILVision/semantic-segmentation-pytorch

# system libs
import os
import argparse

# logging
import logging
from contextlib import redirect_stdout

# object detection
import torch
from torchvision.transforms import ToPILImage

# training
from aprtf.config import cfg
from aprtf.models import ModelBuilder
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

WEIGHT_NAME = 'weights_epoch_{:d}.pth'
WEIGHT_FINAL_NAME = 'weights_final.pth'
WEIGHT_BEST_NAME = 'weights_best.pth'

TRAIN_HEADERS = [TRAIN_EPOCH_NAME, TRAIN_LR_NAME]
VAL_HEADERS = [VAL_AP_NAME, VAL_AR_NAME]
SEP = '\t'


def setup_previous_history(history, cfg):
    # catch up on history
    with open(cfg.MODEL.history, 'r') as f:
        lines = f.readlines()
        # update headers
        cfg.MODEL.history_headers = lines[0].split(SEP)
        # keep track of best epoch
        val_ap_idx = cfg.MODEL.history_headers.index(VAL_AP_NAME)
        for row in f.readlines()[1:]:
            vals = row.split(SEP)
            row_val_ap = float(vals[val_ap_idx])
            if history[BEST_AP_NAME] < row_val_ap:
                epoch_idx = cfg.MODEL.history_headers.index(TRAIN_EPOCH_NAME)
                history[BEST_EPOCH_NAME] = int(epoch_idx)
                history[BEST_AP_NAME] = row_val_ap


def setup_loss_details(train_log, cfg):
    loss_names = [loss_name for loss_name in train_log.meters.keys() if 'loss' in loss_name]
    loss_headers = [TRAIN_NAME + '/' + loss_name for loss_name in loss_names]
    # check same loss headers if part of partial run
    if cfg.TRAIN.start_epoch > 0:
        loss_headers_prev = cfg.MODEL.history_headers[len(TRAIN_HEADERS):-len(VAL_HEADERS)]
        for i in range(len(cfg.MODEL.history_headers)):
            assert loss_headers_prev[i] == loss_headers[i], 'Loss headers have changed!'
    else:
        cfg.MODEL.history_headers = TRAIN_HEADERS + loss_headers + VAL_HEADERS
        with open(cfg.MODEL.history, 'w') as f:
            f.write(SEP.join(cfg.MODEL.history_headers) + '\n')
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

    visualize_results(cfg.VAL.visual, plot_images, gt_bbs, dt_bbs)


def checkpoint(model, history, cfg, epoch):
    logging.info('Saving checkpoint')

    dict_model = model.state_dict()

    # save current model
    if (epoch+1) == cfg.TRAIN.num_epoch:
        weights_fp = os.path.join(cfg.DIR, WEIGHT_FINAL_NAME)
    else:
        weights_fp = os.path.join(cfg.DIR, WEIGHT_NAME.format(epoch))
    torch.save(dict_model, weights_fp)

    # update best model
    if history[BEST_EPOCH_NAME] == epoch:
        torch.save(dict_model, os.path.join(cfg.DIR, WEIGHT_BEST_NAME))

    # update history
    with open(cfg.MODEL.history, 'a') as f:
        stats = SEP.join([str(history[x]) for x in cfg.MODEL.history_headers]) + '\n'
        f.write(stats)

    # delete weights from the previous epoch
    if epoch > 0:
        prev_epoch = epoch - 1
        prev_weight_file = os.path.join(cfg.DIR, WEIGHT_NAME.format(prev_epoch))
        if os.path.exists(prev_weight_file):
            os.remove(prev_weight_file)
            logging.info(f'Previous weights (epoch {prev_epoch}) deleted')


def main(cfg, device):
    model = ModelBuilder.build_detector(arch=cfg.MODEL.arch,
                                        num_classes=cfg.DATASET.num_classes,
                                        weights=cfg.MODEL.weights)
    model.to(device)

    # dataset
    dataset = PedestrianDetectionDataset(cfg.DATASET.list_train, 
                                         transforms=get_transform(True, cfg.DATASET.image_max_size))
    dataset_val = PedestrianDetectionDataset(cfg.DATASET.list_val,
                                             transforms=get_transform(False, cfg.DATASET.image_max_size))

    # dataloaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=cfg.TRAIN.batch_size, shuffle=True, num_workers=cfg.TRAIN.num_workers,
        collate_fn=collate_fn)
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, batch_size=cfg.VAL.batch_size, shuffle=False, num_workers=cfg.VAL.num_workers,
        collate_fn=collate_fn)

    # optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=cfg.TRAIN.lr,
                                momentum=cfg.TRAIN.momentum, 
                                weight_decay=cfg.TRAIN.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=cfg.TRAIN.lr_step_size,
                                                   gamma=cfg.TRAIN.lr_gamma)
    
    # track metrics
    history = {
        BEST_EPOCH_NAME: 0,
        BEST_AP_NAME: -1
    }
    # catch up
    if cfg.TRAIN.start_epoch > 0:
        # catch up on learning rate
        for i in range(0, cfg.TRAIN.start_epoch):
            lr_scheduler.step()
        # catch up on history
        setup_previous_history(history, cfg)

    # training
    for epoch in range(cfg.TRAIN.start_epoch, cfg.TRAIN.num_epoch):
        # early stopping
        if cfg.TRAIN.early_stop < epoch - history[BEST_EPOCH_NAME]:
            logging.info(f'Early stop! No improvement in validation set for {cfg.TRAIN.early_stop} epochs')
            break
        else:
            logging.info(f'Starting Epoch {epoch}')
            
        # train + evaluate
        with redirect_stdout(logging):
            train_log = train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
            eval_log = evaluate(model, data_loader_val, device=device)

        # make loss headers in first epoch of run 
        if epoch == cfg.TRAIN.start_epoch:
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
        visual_evaluate(model, data_loader_val, cfg, device=device)

        # update learning rate
        lr_scheduler.step()

    logging.info('Training Done!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="PyTorch Pedestrian Detection Finetuning"
    )
    parser.add_argument(
        "--cfg",
        default="config/retinanet_resnet50_fpn-pennfudan.yaml",
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

    # make output directory
    if not os.path.isdir(cfg.DIR):
        os.makedirs(cfg.DIR)
    elif cfg.TRAIN.start_epoch == 0:
        # starting from scratch
        for f in os.listdir(cfg.DIR):
            os.remove(os.path.join(cfg.DIR,f))

    # make config 
    config_fp = os.path.join(cfg.DIR, cfg.MODEL.config_name)
    if not os.path.exists(config_fp):
        with open(config_fp, 'w') as f:
            f.write(str(cfg))

    # start from checkpoint
    cfg.MODEL.history = os.path.join(cfg.DIR, cfg.MODEL.history_name)
    cfg.MODEL.weights = ""
    if cfg.TRAIN.start_epoch > 0:
        cfg.MODEL.weights = os.path.join(cfg.DIR, WEIGHT_NAME.format(cfg.TRAIN.start_epoch-1))
        assert os.path.exists(cfg.MODEL.weights), "weight checkpoint does not exist!"
        assert os.path.exists(cfg.MODEL.history), "history checkpoint does not exist!"

    # make visual filepath
    cfg.VAL.visual = os.path.join(cfg.DIR, cfg.VAL.visual_name)

    # setup logger
    log_fp = os.path.join(cfg.DIR, cfg.MODEL.log_name)
    if not os.path.exists(log_fp):
        open(log_fp, 'a').close()
    logging.basicConfig(level=logging.INFO,
                        format='[%(asctime)s %(levelname)s %(filename)s] %(message)s',
                        datefmt='%H:%M:%S',
                        handlers=[logging.FileHandler(log_fp)])
    # for redirecting stdout
    logging.write = lambda msg: logging.info(msg) if msg != '\n' else None

    # log details
    logging.info("Loaded configuration file: {}".format(args.cfg))
    logging.info("Running with config:\n{}".format(cfg))
    logging.info("Outputting to: {}".format(cfg.DIR))

    # random seed
    torch.manual_seed(cfg.TRAIN.seed)

    # train on the GPU or on the CPU, if a GPU is not available
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    main(cfg, device)
