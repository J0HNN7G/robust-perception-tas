# robust-perception-tas
Robustness Testing Framework for Image Models in Autonomous Systems

##  Create Conda Environment
```
conda create -n <env_name>
conda activate <name>

# pytorch (change to desired version)
conda install pytorch==2.0.0 torchvision==0.15.0 pytorch-cuda=11.7 -c pytorch -c nvidia

# data processing
conda install numpy matplotlib Pillow

# config and metrics (not on conda)
pip install yacs cython pycocotools -upgrade-strategy only-if-needed
```

## Training

1. Setup dataset.
```
cd <robust-perception-tas_directory>
python setupPennFudan.py
```
Custom dataset files are expected to be formatted as follows:
```
{'image': <image_filepath>, 'annotations': [[xmin, ymin, xmax, ymax],...]}
...
```


2. Make configuration YAML file. 

Example `retinanet_resnet50_fpn-pennfudan.yaml`:
```
DATASET:
  root_dataset: "data/sets/PennFudanPed"
  list_train: "data/sets/PennFudanPed/train_PennFudanPed.odgt"
  list_val: "data/sets/PennFudanPed/val_PennFudanPed.odgt"
  num_classes: 2
  image_max_size: 720

MODEL:
  arch: "retinanet_resnet50_fpn"

TRAIN:
  batch_size: 32
  num_epoch: 10
  start_epoch: 0
  optim: "SGD"
  momentum: 0.9
  weight_decay: 0.0005
  lr: 0.005
  lr_step_size: 3
  lr_gamma: 0.1
  early_stop: 2
  num_workers: 4
  disp_iter: 10
  seed: 42

VAL:
  batch_size: 1
  checkpoint: "weights_best.pth"

TEST:
  checkpoint: "weights_best.pth"
  result: "./results"

DIR: "ckpt/retinanet_resnet50_fpn-pennfudan"
```

3. Run the training
```
python train.py --cfg config/<config_file>
```

4. Results are stored at directory specified by `DIR` in configuration file. By default your directory will be set up as follows:
```
.
├── build                   # Compiled files (alternatively `dist`)
├── 
├── weights_best.pth        # checkpoint with best validation mAP
├── weigths_epoch_<n>.pth   # last checkpoint whilst running
├── weights_final.pth       # final checkpoint if run finished
├── history.tsv             # training and validation metrics history
├── config.yaml             # configuration file copy
└── log.txt                 # model training logs
```
