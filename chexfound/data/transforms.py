# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

from typing import Sequence

import torch
from torchvision import transforms
from torchvision.transforms import functional as F

import numpy as np

class GaussianBlur(transforms.RandomApply):
    """
    Apply Gaussian Blur to the PIL image.
    """

    def __init__(self, *, p: float = 0.5, radius_min: float = 0.1, radius_max: float = 2.0):
        # NOTE: torchvision is applying 1 - probability to return the original image
        keep_p = 1 - p
        transform = transforms.GaussianBlur(kernel_size=9, sigma=(radius_min, radius_max))
        super().__init__(transforms=[transform], p=keep_p)


class MaybeToTensor(transforms.ToTensor):
    """
    Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor, or keep as is if already a tensor.
    """

    def __call__(self, pic):
        """
        Args:
            pic (PIL Image, numpy.ndarray or torch.tensor): Image to be converted to tensor.
        Returns:
            Tensor: Converted image.
        """
        if isinstance(pic, torch.Tensor):
            return pic
        return super().__call__(pic)


class RescaleImage:
    def __call__(self, image):
        if isinstance(image, np.ndarray):
            # Convert to tensor
            image = torch.from_numpy(image)
        elif torch.is_tensor(image):
            pass
        else:
            raise TypeError("Input should be of type numpy.ndarray or torch.Tensor")

        # Rescale the tensor to [0, 1]
        min_val = image.reshape(image.shape[0], -1).min(dim=1)[0].reshape(-1, 1, 1)
        max_val = image.reshape(image.shape[0], -1).max(dim=1)[0].reshape(-1, 1, 1)
        return (image - min_val) / (max_val - min_val)


class RandomRot90:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image):
        if not np.random.uniform() < self.p:
            pass
        else:
            k_choices = [1, 2, 3]
            k = np.random.choice(k_choices)
            image = image.rotate(90*k, expand=True)
        return image


# Use timm's names
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


def make_normalize_transform(
    mean: Sequence[float] = IMAGENET_DEFAULT_MEAN,
    std: Sequence[float] = IMAGENET_DEFAULT_STD,
) -> transforms.Normalize:
    return transforms.Normalize(mean=mean, std=std)


# This roughly matches torchvision's preset for classification training:
#   https://github.com/pytorch/vision/blob/main/references/classification/presets.py#L6-L44
def make_classification_train_transform(
    *,
    crop_size: int = 224,
    interpolation=transforms.InterpolationMode.BICUBIC,
    hflip_prob: float = 0.5,
    mean: Sequence[float] = IMAGENET_DEFAULT_MEAN,
    std: Sequence[float] = IMAGENET_DEFAULT_STD,
    rot90: bool = False,
):
    transforms_list = [transforms.RandomResizedCrop(crop_size, scale=(0.75, 1), interpolation=interpolation)]
    if hflip_prob > 0.0:
        transforms_list.append(transforms.RandomHorizontalFlip(hflip_prob))
    if rot90:
        transforms_list.append(RandomRot90(p=0.5))
    transforms_list.extend(
        [
            MaybeToTensor(),
            RescaleImage(),
            make_normalize_transform(mean=mean, std=std),
        ]
    )
    return transforms.Compose(transforms_list)


# This matches (roughly) torchvision's preset for classification evaluation:
#   https://github.com/pytorch/vision/blob/main/references/classification/presets.py#L47-L69
def make_classification_eval_transform(
    *,
    resize_size: int = 256,
    interpolation=transforms.InterpolationMode.BICUBIC,
    crop_size: int = 224,
    mean: Sequence[float] = IMAGENET_DEFAULT_MEAN,
    std: Sequence[float] = IMAGENET_DEFAULT_STD,
    flip: bool = False,
) -> transforms.Compose:
    transforms_list = [
        transforms.Resize(resize_size, interpolation=interpolation),
        transforms.CenterCrop(crop_size),
        MaybeToTensor(),
        RescaleImage(),
        make_normalize_transform(mean=mean, std=std),
    ]
    if flip:
        transforms_list.append(transforms.RandomHorizontalFlip(p=1))
    return transforms.Compose(transforms_list)

def make_classification_eval_transform_nocrop(
    *,
    target_size: int = 256,
    mean: Sequence[float] = IMAGENET_DEFAULT_MEAN,
    std:  Sequence[float] = IMAGENET_DEFAULT_STD,
    flip: bool = False,
) -> transforms.Compose:
    interpolation=transforms.InterpolationMode.BICUBIC
    def resize_and_letterbox(img):
        # 1) Skalierung: kleinere Kante → target_size
        img = resize_long_edge(img, target_size, interpolation)
        # 2) Padding: zentriert auf target_size×target_size auffüllen
        w, h = img.size
        pad_w = max(target_size - w, 0)
        pad_h = max(target_size - h, 0)
        # (links, oben, rechts, unten)
        padding = (pad_w // 2, pad_h // 2, pad_w - pad_w // 2, pad_h - pad_h // 2)
        return F.pad(img, padding, fill=0)
    
    transforms_list = [
        transforms.Lambda(resize_and_letterbox),
        MaybeToTensor(),
        RescaleImage(),
        make_normalize_transform(mean=mean, std=std),
    ]
    if flip:
        # horizontal flip nach Padding/Crop
        transforms_list.append(T.RandomHorizontalFlip(p=1.0))
    return transforms.Compose(transforms_list)

def resize_long_edge(img, size, interpolation):
    w, h = img.size
    # Größere Kante anpassen
    if h > w:
        new_h = size
        new_w = int(w * size / h)
    else:
        new_w = size
        new_h = int(h * size / w)
    return F.resize(img, (new_h, new_w), interpolation)