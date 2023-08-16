# code altered from MIT semantic segmentation repository
# https://github.com/CSAILVision/semantic-segmentation-pytorch

from yacs.config import CfgNode as CN

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()
_C.DIR = "ckpt/fasterrcnn_resnet50_fpn-pennfudan"

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASET = CN()
_C.DATASET.root_dataset = "data/sets/PennFudanPed"
_C.DATASET.list_train = "data/sets/PennFudanPed"
_C.DATASET.list_val = "data/sets/PennFudanPed"
_C.DATASET.num_classes = 2
# maximum length along any dimension
_C.DATASET.image_max_size = 720

# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
_C.MODEL = CN()
# name of model architecture
_C.MODEL.arch = "fasterrcnn_resnet50_fpn"
# history name
_C.MODEL.history_name = "history.tsv"
# log name
_C.MODEL.log_name = "log.txt"
# config name
_C.MODEL.config_name = "config.yaml"

# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------
_C.TRAIN = CN()
_C.TRAIN.batch_size = 32
# epochs to train for
_C.TRAIN.num_epoch = 10
# epoch to start training. useful if continue from a checkpoint
_C.TRAIN.start_epoch = 0
# optimizer
_C.TRAIN.optim = "SGD"
# initial learning rate
_C.TRAIN.lr = 0.005
# period of learning rate decay
_C.TRAIN.lr_step_size = 3
# multiplicative factor of learning rate decay
_C.TRAIN.lr_gamma = 0.1
# momentum for sgd, beta1 for adam
_C.TRAIN.momentum = 0.9
# weights regularizer
_C.TRAIN.weight_decay = 0.0005
# number of data loading workers
_C.TRAIN.num_workers = 4
# early stopping if no improvements in this many epochs
_C.TRAIN.early_stop = 2
# frequency to display
_C.TRAIN.disp_iter = 20
# manual seed
_C.TRAIN.seed = ''


# -----------------------------------------------------------------------------
# Validation
# -----------------------------------------------------------------------------
_C.VAL = CN()
# currently only supports 1
_C.VAL.batch_size = 1
# number of data loading workers
_C.VAL.num_workers = 4
# the checkpoint to evaluate on
_C.VAL.checkpoint = f"weights_best.pth"
# visual name
_C.VAL.visual_name = "visual.png"

# -----------------------------------------------------------------------------
# Testing
# -----------------------------------------------------------------------------
_C.TEST = CN()
# the checkpoint to test on
_C.TEST.checkpoint = "weights_best.pth"
# folder to output visualization results
_C.TEST.result = "./results"