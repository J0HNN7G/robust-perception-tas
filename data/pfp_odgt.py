"""Generate ODGT label files for PennFudanPed dataset"""
# general
import os
import json
import argparse

# boxes
from PIL import Image
import numpy as np

# progress bar
from tqdm import tqdm

# constants
TRAIN_NAME = 'train'
VAL_NAME = 'val'
ODGT_NAME = 'PennFudanPed.odgt'
IMG_DIR_NAME = "PNGImages"
MASK_DIR_NAME = "PedMasks"
DATASET_NAME = 'PennFudanPed'


def indices2odgt(odgt_fp, dir_p, indices, img_fps, mask_fps):
    """
    Generate ODGT label files for PennFudanPed dataset using given indices and file paths.

    Parameters:
    - odgt_fp (str): File path to save the ODGT label file.
    - dir_p (str): Directory path to the dataset.
    - indices (list): List of indices to process.
    - img_fps (list): List of image file paths.
    - mask_fps (list): List of mask file paths.

    Returns:
    None
    """
    for idx in tqdm(indices):
        mask_path = os.path.join(dir_p, MASK_DIR_NAME, mask_fps[idx]) 
        mask = Image.open(mask_path)

        mask = np.array(mask)
        # instances are encoded as different colors
        obj_ids = np.unique(mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]

        # split the color-encoded mask into a set
        # of binary masks
        masks = mask == obj_ids[:, None, None]

        # get bounding box coordinates for each mask
        boxes = []
        for i in range(len(obj_ids)):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            box =  [xmin, ymin, xmax, ymax]
            boxes.append([int(x) for x in box])

        sample = {
            'image': os.path.join(dir_p, IMG_DIR_NAME, img_fps[idx]),
            'annotations': boxes
        }  

        with open(odgt_fp, 'a') as f:
            json.dump(sample, f)
            f.write('\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Penn-Fudan Pedestrian Dataset setup"
    )
    parser.add_argument(
        "--dir",
        required=True,
        metavar="PATH",
        help="absolute path to intended PennFudan dataset directory",
        type=str,
    )
    parser.add_argument(
        "--tfrac",
        default=0.8,
        metavar="FLOAT",
        help="fraction of samples to put in training set vs validation set",
        type=float,
    )
    args = parser.parse_args()
    if not os.path.exists(args.dir):
        raise ValueError('Cannot find data directory')
    if not 0 <= args.tfrac <= 1:
        raise ValueError('train fraction out of range [0,1]!')

    print(f'Starting odgt file creation')
    dataset_dir_p = os.path.join(args.dir, DATASET_NAME)
    img_fps = list(sorted(os.listdir(os.path.join(dataset_dir_p, IMG_DIR_NAME))))
    mask_fps = list(sorted(os.listdir(os.path.join(dataset_dir_p, MASK_DIR_NAME))))
    assert len(img_fps) == len(mask_fps), 'Different number of images to annotations!'

    indices = np.random.permutation(len(img_fps)).tolist()
    limit = int(len(img_fps) * args.tfrac) 

    print(f'Train indexing')
    odgt_fp_train = os.path.join(dataset_dir_p, f'{TRAIN_NAME}_{ODGT_NAME}')
    if os.path.exists(odgt_fp_train):
        print('Train indexing already done!')
    else:
        open(odgt_fp_train, 'w').close() 
        indices2odgt(odgt_fp_train, dataset_dir_p, indices[:limit], img_fps, mask_fps)
        print(f'Train file saved at: {odgt_fp_train}')

    print(f'Validation indexing')
    odgt_fp_val = os.path.join(dataset_dir_p, f'{VAL_NAME}_{ODGT_NAME}')
    if os.path.exists(odgt_fp_val):
        print('Validation indexing already done!')
    else:
        open(odgt_fp_val, 'w').close() 
        indices2odgt(odgt_fp_val, dataset_dir_p, indices[limit:], img_fps, mask_fps)
        print(f'Validation file saved at: {odgt_fp_val}')





    

