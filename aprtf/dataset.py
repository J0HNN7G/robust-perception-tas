# code altered from TorchVision 0.3 Object Detection Finetuning Tutorial
# http://pytorch.org/tutorials/intermediate/torchvision_tutorial.html

# files
import os

# image
import numpy as np
from PIL import Image
import torch

# bounding boxes
from torchvision.ops import clip_boxes_to_image, remove_small_boxes
from aprtf.references import transforms as T


def get_transform(train):
    transforms = []
    transforms.append(T.PILToTensor())
    transforms.append(T.ConvertImageDtype(torch.float))
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


class PedestrianDetectionDataset(object):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, "PNGImages"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "PedMasks"))))

    def __getitem__(self, idx):
        # load images and masks
        img_path = os.path.join(self.root, "PNGImages", self.imgs[idx])
        mask_path = os.path.join(self.root, "PedMasks", self.masks[idx])
        img = Image.open(img_path).convert("RGB")
        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance
        # with 0 being background
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
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)


class PedestrianDetectionDataset2(torch.utils.data.Dataset):
    def __init__(self, images, boxes, transforms, bb_min_len=40):
        assert len(images) == len(boxes)
        self.images = images
        self.boxes = boxes
        self.transforms = transforms
        self.bb_min_len = bb_min_len

    def __getitem__(self, idx):
        # image
        img_path = self.images[idx]
        img = Image.open(img_path).convert("RGB")
        
        # boxes
        bbs = self.boxes[idx]
        bbs = torch.as_tensor(bbs, dtype=torch.float32)
        bbs = torch.reshape(bbs, (-1,4))
        # clip all boxes to image dims
        bbs = clip_boxes_to_image(bbs, (img.size[1], img.size[0]))
        # remove boxes with too small size
        #bbs = bbs[remove_small_boxes(bbs, self.bb_min_len)]
        
        num_objs = len(bbs)
        target = {}
        target["boxes"] = bbs
        target["labels"] = torch.ones((num_objs,), dtype=torch.int64)
        target["image_id"] = torch.tensor([idx])
        target["area"] = (bbs[..., 3] - bbs[..., 1]) * (bbs[..., 2] - bbs[..., 0])
        target["iscrowd"] = torch.zeros((num_objs,), dtype=torch.int64)

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.images)