# general
import os
import json
import argparse

# boxes
from PIL import Image
import numpy as np

# constants
TRAIN_NAME = 'train'
VAL_NAME = 'val'
ODGT_NAME = 'PennFudanPed.odgt'
IMG_DIR_NAME = "PNGImages"
MASK_DIR_NAME = "PedMasks"


def indices2odgt(odgt_fp, indices, img_fps, mask_fps):
    for idx in indices:
        mask_path = os.path.join(args.dir, MASK_DIR_NAME, mask_fps[idx]) 
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
            'image': os.path.join(args.dir, IMG_DIR_NAME, img_fps[idx]),
            'annotations': boxes
        }  

        with open(odgt_fp, 'a') as f:
            json.dump(sample, f)
            f.write('\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="PennFudanPed ODGT setup"
    )
    parser.add_argument(
        "--dir",
        default="data/sets/PennFudanPed",
        metavar="DIRECTORY",
        help="path to PennFudanPed dataset directory",
        type=str,
    )
    parser.add_argument(
        "--tfrac",
        default=0.8,
        metavar="VALUE",
        help="fraction of samples to put in training set vs validation set",
        type=float,
    )
    args = parser.parse_args()

    print(f'Starting odgt file creation')

    img_fps = list(sorted(os.listdir(os.path.join(args.dir, IMG_DIR_NAME))))
    mask_fps = list(sorted(os.listdir(os.path.join(args.dir, MASK_DIR_NAME))))
    assert len(img_fps) == len(mask_fps), 'Different number of images to annotations!'

    indices = np.random.permutation(len(img_fps)).tolist()
    limit = int(len(img_fps) * args.tfrac) 

    odgt_fp_train = os.path.join(args.dir, f'{TRAIN_NAME}_{ODGT_NAME}')
    open(odgt_fp_train, 'w').close() 
    indices2odgt(odgt_fp_train, indices[:limit], img_fps, mask_fps)
    print(f'Train file saved at: {odgt_fp_train}')

    odgt_fp_val = os.path.join(args.dir, f'{VAL_NAME}_{ODGT_NAME}')
    open(odgt_fp_val, 'w').close() 
    indices2odgt(odgt_fp_val, indices[limit:], img_fps, mask_fps)
    print(f'Validation file saved at: {odgt_fp_val}')





    

