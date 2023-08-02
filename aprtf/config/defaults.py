# code altered from MIT semantic segmentation repository
# https://github.com/CSAILVision/semantic-segmentation-pytorch

from yacs.config import CfgNode as CN

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()
_C.DIR = "ckpt/fasterrcnn_resnet50_fpn_v2-pennfudan"

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASET = CN()
_C.DATASET.root_dataset = "data/sets/PennFudanPed"
_C.DATASET.list_train = "data/sets/PennFudanPed"
_C.DATASET.list_val = "data/sets/PennFudanPed"
_C.DATASET.num_classes = 2

# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
_C.MODEL = CN()
# name of model architecture
_C.MODEL.arch = "fasterrcnn_resnet50_fpn_v2"
# filepath to pre-trained weights
_C.MODEL.weights = ""
# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------
_C.TRAIN = CN()
_C.TRAIN.batch_size = 2
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
_C.TRAIN.early_stop = 1
# frequency to display
_C.TRAIN.disp_iter = 10
# manual seed
_C.TRAIN.seed = 42

# -----------------------------------------------------------------------------
# Validation
# -----------------------------------------------------------------------------
_C.VAL = CN()
# currently only supports 1
_C.VAL.batch_size = 1
# the checkpoint to evaluate on
_C.VAL.checkpoint = f"best_checkpoint.pth"

# -----------------------------------------------------------------------------
# Testing
# -----------------------------------------------------------------------------
_C.TEST = CN()
# currently only supports 1
_C.TEST.batch_size = 1
# the checkpoint to test on
_C.TEST.checkpoint = "best_checkpoint.pth"
# folder to output visualization results
_C.TEST.result = "./results"