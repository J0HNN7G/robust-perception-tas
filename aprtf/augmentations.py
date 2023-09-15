"""Augmentation methods"""


class TransformAugmentationCompose:
    def __init__(self, transforms, augmentations):
        self.transforms = transforms
        self.augmentations = augmentations

    def __call__(self, image, target):
        image, target = self.transforms(image, target)
        image = self.augmentations(image)
        return image, target