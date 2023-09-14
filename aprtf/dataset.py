"""Load pedestrian detection dataset"""
# code altered from TorchVision 0.3 Object Detection Finetuning Tutorial
# http://pytorch.org/tutorials/intermediate/torchvision_tutorial.html

# files
import json

# image
import torch
from PIL import Image

# bounding boxes
from aprtf.torchvision_detection import transforms as T


def get_transform(train):
    """
    Get a composition of image transformation functions.

    Args:
        train (bool): A boolean flag indicating whether the transformations are for training.

    Returns:
        torchvision.transforms.Compose: A composition of transformation functions.
    """
    transforms = []
    transforms.append(T.PILToTensor())
    transforms.append(T.ConvertImageDtype(torch.float))
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


class PedestrianDetectionDataset(torch.utils.data.Dataset):
    """Custom dataset for pedestrian detection."""
    def __init__(self, odgt, transforms):
        """
        Initializes the dataset.

        Args:
            odgt (str or list): Either a JSON file path or a list of data samples (annotations).
            transforms (callable): A list of image transformation functions.
        """
        if isinstance(odgt, str):
            self.list_sample = [json.loads(x.rstrip()) for x in open(odgt, 'r')]
        elif isinstance(odgt, list):
            self.list_sample = odgt
        else:
            raise ValueError('Undefined parse for ODGT type!')
        
        self.transforms = transforms

    def __getitem__(self, idx):
        """
        Retrieves a data sample from the dataset.

        Args:
            idx (int): The index of the data sample to retrieve.

        Returns:
            tuple: A tuple containing the image and target information 
            (bounding boxes, labels, image ID, area, and iscrowd).
        """
        # image
        img_path = self.list_sample[idx]['image']
        img = Image.open(img_path).convert("RGB")

        # boxes
        bbs = self.list_sample[idx]['annotations']
        bbs = torch.as_tensor(bbs, dtype=torch.float)
        bbs = torch.reshape(bbs, (-1,4))
        
        num_objs = len(bbs)
        target = {}
        target["boxes"] = bbs
        # there is only one class
        target["labels"] = torch.ones((num_objs,), dtype=torch.int64)
        target["image_id"] = torch.tensor([idx])
        target["area"] = (bbs[..., 3] - bbs[..., 1]) * (bbs[..., 2] - bbs[..., 0])
        # suppose all instances are not crowd
        target["iscrowd"] = torch.zeros((num_objs,), dtype=torch.int64)

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        """
        Returns the number of data samples in the dataset.
        """
        return len(self.list_sample)