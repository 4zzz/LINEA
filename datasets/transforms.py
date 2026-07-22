# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Transforms and data augmentation for both image + bbox.
"""
import random

import PIL
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
import numbers

import numpy as np


def _endpoint_swap_mask(lines):
    return torch.logical_or(
        lines[..., 0] > lines[..., 2],
        torch.logical_and(
            lines[..., 0] == lines[..., 2],
            lines[..., 1] > lines[..., 3],
        ),
    )


def _swap_line_endpoints(lines, lines3d, swap_mask):
    lines = lines.clone()
    lines[swap_mask] = lines[swap_mask][:, [2, 3, 0, 1]]
    if lines3d is None:
        return lines, None

    lines3d = lines3d.view(-1, 2, 3).clone()
    lines3d[swap_mask] = lines3d[swap_mask][:, [1, 0], :]
    return lines, lines3d.reshape(-1, 6)


def _clip_lines3d_to_projected_lines(lines3d, source_lines2d, clipped_lines2d):
    points = lines3d.view(-1, 2, 3)
    source_uv = source_lines2d.view(-1, 2, 2)
    clipped_uv = clipped_lines2d.view(-1, 2, 2)

    uv_delta = source_uv[:, 1] - source_uv[:, 0]
    uv_norm_sq = uv_delta.square().sum(dim=-1).clamp_min(1e-12)
    screen_t = (
        (clipped_uv - source_uv[:, :1]) * uv_delta[:, None]
    ).sum(dim=-1) / uv_norm_sq[:, None]
    screen_t = screen_t.clamp(0.0, 1.0)

    z0 = points[:, 0, 2:3]
    z1 = points[:, 1, 2:3]
    denominator = (1.0 - screen_t) * z1 + screen_t * z0
    denominator = torch.where(
        denominator.abs() < 1e-12,
        torch.full_like(denominator, 1e-12),
        denominator,
    )
    line_t = screen_t * z0 / denominator
    clipped_points = points[:, :1] + line_t[..., None] * (points[:, 1:] - points[:, :1])
    return clipped_points.reshape(-1, 6)


def crop(image, target, region):
    cropped_image = F.crop(image, *region)

    target = target.copy()
    i, j, h, w = region

    # should we do something wrt the original size?
    target["size"] = torch.tensor([h, w])

    if "camera_K" in target:
        camera_k = target["camera_K"].clone()
        camera_k[0, 2] -= j
        camera_k[1, 2] -= i
        target["camera_K"] = camera_k

    fields = ["labels", "area", "iscrowd"]

    if 'lmap' in target:
        cropped_lmaps = []
        for lmap, downsampling in zip(target['lmap'], [8, 16, 32]):
            cropped_lmap = F.crop(lmap, i//downsampling, j//downsampling, h//downsampling, w//downsampling)
            cropped_lmaps.append(cropped_lmap)
        target['lmap'] = cropped_lmaps

    if "lines" in target:
        lines = target["lines"]
        cropped_lines = lines - torch.as_tensor([j, i, j, i])
        
        eps = 1e-12

        # In dataset, we assume the left point has smaller x coord
        remove_x_min = cropped_lines[:, 2] < 0
        remove_x_max = cropped_lines[:, 0] > w
        remove_x = torch.logical_or(remove_x_min, remove_x_max)
        keep_x = ~remove_x

        # there is no assumption on y, so remove lines that have both y coord out of bound
        remove_y_min = torch.logical_and(cropped_lines[:, 1] < 0, cropped_lines[:, 3] < 0)
        remove_y_max = torch.logical_and(cropped_lines[:, 1] > h, cropped_lines[:, 3] > h)
        remove_y = torch.logical_or(remove_y_min, remove_y_max)
        keep_y = ~remove_y

        keep = torch.logical_and(keep_x, keep_y)
        cropped_lines = cropped_lines[keep]
        cropped_lines3d = target.get("lines3d")
        if cropped_lines3d is not None:
            cropped_lines3d = cropped_lines3d[keep]
        clamped_lines = torch.zeros_like(cropped_lines)

        for i,line in enumerate(cropped_lines):
            x1, y1, x2, y2 = line
            slope = (y2 - y1) / (x2 - x1 + eps)
            if x1 < 0:
                x1 = 0
                y1 = y2 + (x1 - x2) * slope
            if y1 < 0 or y1 > h-1:
                y1 = 0 if y1 < 0 else h -1 
                x1 = x2 - (y2 - y1) / slope
            if x2 >= w:
                x2 = w - 1
                y2 = y1 + (x2 - x1) * slope
            if y2<0 or y2 > h-1:
                y2 = 0 if y2 < 0 else h -1 
                x2 = x1 + (y2 - y1) / slope

            clamped_lines[i, :] = torch.tensor([x1, y1, x2, y2])

        # remove some noisy points 
        keep_real_lines = (clamped_lines[:, :2] - clamped_lines[:, 2:]).norm(dim=1) > 10
        
        target["lines"] = clamped_lines[keep_real_lines]
        if cropped_lines3d is not None:
            target["lines3d"] = _clip_lines3d_to_projected_lines(
                cropped_lines3d,
                cropped_lines,
                clamped_lines,
            )[keep_real_lines]

    for field in fields:
        target[field] = target[field][keep][keep_real_lines]

    return cropped_image, target


def hflip(image, target):
    flipped_image = F.hflip(image)

    w, h = image.size

    target = target.copy()
    if "lines" in target:
        lines = target["lines"]   
        lines = lines[:, [2, 3, 0, 1]] * torch.as_tensor([-1, 1, -1, 1]) + torch.as_tensor([w - 1, 0, w - 1, 0])
        target["lines"] = lines
        if "lines3d" in target:
            target["lines3d"] = target["lines3d"].view(-1, 2, 3)[:, [1, 0], :].reshape(-1, 6)
    if "camera_K" in target:
        camera_k = target["camera_K"].clone()
        camera_k[0, 0] = -camera_k[0, 0]
        camera_k[0, 2] = w - 1 - camera_k[0, 2]
        target["camera_K"] = camera_k

    if 'lmap' in target:
        flipped_lmaps = []
        for lmap in target['lmap']:
            flipped_lmap = F.hflip(lmap)
            flipped_lmaps.append(flipped_lmap)
        target['lmap'] = flipped_lmaps

    return flipped_image, target

def vflip(image, target):
    flipped_image = F.vflip(image)

    w, h = image.size

    target = target.copy()

    if "lines" in target:
        lines = target["lines"]

        # in dataset, we assume if two points with same x coord, we assume first point is the upper point
        lines = lines * torch.as_tensor([1, -1, 1, -1]) + torch.as_tensor([0, h - 1, 0, h - 1])
        vertical_line_idx = (lines[:, 0] == lines[:, 2])
        lines[vertical_line_idx] = torch.index_select(lines[vertical_line_idx], 1, torch.tensor([2,3,0,1]))
        target["lines"] = lines
        if "lines3d" in target:
            lines3d = target["lines3d"].view(-1, 2, 3)
            lines3d[vertical_line_idx] = lines3d[vertical_line_idx][:, [1, 0], :]
            target["lines3d"] = lines3d.reshape(-1, 6)
    if "camera_K" in target:
        camera_k = target["camera_K"].clone()
        camera_k[1, 1] = -camera_k[1, 1]
        camera_k[1, 2] = h - 1 - camera_k[1, 2]
        target["camera_K"] = camera_k

    if 'lmap' in target:
        flipped_lmaps = []
        for lmap in target['lmap']:
            flipped_lmap = F.vflip(lmap)
            flipped_lmaps.append(flipped_lmap)
        target['lmap'] = flipped_lmaps

    return flipped_image, target
    
def resize(image, target, size, max_size=None):
    # size can be min_size (scalar) or (w, h) tuple

    def get_size_with_aspect_ratio(image_size, size, max_size=None):
        w, h = image_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (oh, ow)

    def get_size(image_size, size, max_size=None):
        if isinstance(size, (list, tuple)):
            return size[::-1]
        else:
            return get_size_with_aspect_ratio(image_size, size, max_size)

    size = get_size(image.size, size, max_size)
    rescaled_image = F.resize(image, size)

    if target is None:
        return rescaled_image, None

    ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(rescaled_image.size, image.size))
    ratio_width, ratio_height = ratios

    target = target.copy()
    if "lines" in target:
        lines = target["lines"]
        scaled_lines = lines * torch.as_tensor([ratio_width, ratio_height, ratio_width, ratio_height])
        target["lines"] = scaled_lines
    if "camera_K" in target:
        camera_k = target["camera_K"].clone()
        camera_k[0, 0] *= ratio_width
        camera_k[0, 2] *= ratio_width
        camera_k[1, 1] *= ratio_height
        camera_k[1, 2] *= ratio_height
        target["camera_K"] = camera_k

    if 'lmap' in target:
        resize_lmaps = []
        for lmap, downsampling in zip(target['lmap'], [8, 16, 32]):
            resize_lmap = F.resize(lmap, (size[0]//downsampling, size[1]//downsampling))
            resize_lmaps.append(resize_lmap)
        target['lmap'] = resize_lmaps

    h, w = size
    target["size"] = torch.tensor([h, w])
    return rescaled_image, target


def pad(image, target, padding):
    # assumes that we only pad on the bottom right corners
    padded_image = F.pad(image, (0, 0, padding[0], padding[1]))
    if target is None:
        return padded_image, None
    target = target.copy()
    # should we do something wrt the original size?
    target["size"] = torch.tensor(padded_image.size[::-1])
    if "masks" in target:
        target['masks'] = torch.nn.functional.pad(target['masks'], (0, padding[0], 0, padding[1]))
    return padded_image, target

def rotation(image, target, rotation_type):
    """
    image: Tensor (1, C, H, W)
    target: Dict that contains the lines (n, 4)
    rotation_type:
        0: No rotation
        1: Rotate 90 deg
        2: Rotate -90 deg  
    """
    rotation = {0: 0, 1: 90, 2: -90}
    w, h = image.size
    rotated_image = F.rotate(image, rotation[rotation_type])

    target = target.copy()

    if rotation_type == 0:
        target["size"] = torch.tensor([h, w])
        return rotated_image, target

    if rotation_type == 1:
        if "lines" in target:
            lines = target["lines"]
            rotated_lines = lines[..., [1, 0, 3, 2]]
            rotated_lines[..., [1, 3]] = w - 1 - rotated_lines[..., [1, 3]]
        image_transform = torch.tensor(
            [[0.0, 1.0, 0.0], [-1.0, 0.0, w - 1.0], [0.0, 0.0, 1.0]]
        )
    elif rotation_type == 2:
        if "lines" in target:
            lines = target["lines"]
            rotated_lines = lines[..., [1, 0, 3, 2]]
            rotated_lines[..., [0, 2]] = h - 1 - rotated_lines[..., [0, 2]]
        image_transform = torch.tensor(
            [[0.0, -1.0, h - 1.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
        )

    lines3d = target.get('lines3d')
    swap_mask = _endpoint_swap_mask(rotated_lines)
    target['lines'], swapped_lines3d = _swap_line_endpoints(rotated_lines, lines3d, swap_mask)
    if swapped_lines3d is not None:
        target['lines3d'] = swapped_lines3d

    if 'camera_K' in target:
        image_transform = image_transform.to(target['camera_K'])
        target['camera_K'] = image_transform @ target['camera_K']

    if 'lmap' in target:
        rotated_lmap = F.rotate(target['lmap'], rotation[rotation_type])
        target['lmap'] = rotated_lmap

    target["size"] = torch.tensor([w, h])

    return rotated_image, target

class Rotation(object):
    def __init__(self):
        self.rotation_type = [0, 0, 0, 1, 2]

    def __call__(self, img, target):
        rotation_type = np.random.choice(self.rotation_type)
        return rotation(img, target, rotation_type)

class ResizeDebug(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, target):
        return resize(img, target, self.size)


class RandomCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, target):
        region = T.RandomCrop.get_params(img, self.size)
        return crop(img, target, region)


class RandomSizeCrop(object):
    def __init__(self, min_size: int, max_size: int):
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, img: PIL.Image.Image, target: dict):
        w = random.randint(self.min_size, min(img.width, self.max_size))
        h = random.randint(self.min_size, min(img.height, self.max_size))
        region = T.RandomCrop.get_params(img, [h, w])
        return crop(img, target, region)


class CenterCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, target):
        image_width, image_height = img.size
        crop_height, crop_width = self.size
        crop_top = int(round((image_height - crop_height) / 2.))
        crop_left = int(round((image_width - crop_width) / 2.))
        return crop(img, target, (crop_top, crop_left, crop_height, crop_width))


class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            return hflip(img, target)
        return img, target


class RandomVerticalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            return vflip(img, target)
        return img, target
        
        
class RandomResize(object):
    def __init__(self, sizes, max_size=None):
        assert isinstance(sizes, (list, tuple))
        self.sizes = sizes
        self.max_size = max_size

    def __call__(self, img, target=None):
        size = random.choice(self.sizes)
        return resize(img, target, size, self.max_size)


class RandomPad(object):
    def __init__(self, max_pad):
        self.max_pad = max_pad

    def __call__(self, img, target):
        pad_x = random.randint(0, self.max_pad)
        pad_y = random.randint(0, self.max_pad)
        return pad(img, target, (pad_x, pad_y))


class RandomSelect(object):
    """
    Randomly selects between transforms1 and transforms2,
    with probability p for transforms1 and (1 - p) for transforms2
    """
    def __init__(self, transforms1, transforms2, p=0.5):
        self.transforms1 = transforms1
        self.transforms2 = transforms2
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            return self.transforms1(img, target)
        return self.transforms2(img, target)


class ToTensor(object):
    def __call__(self, img, target):
        return F.to_tensor(img), target


class RandomErasing(object):

    def __init__(self, *args, **kwargs):
        self.eraser = T.RandomErasing(*args, **kwargs)

    def __call__(self, img, target):
        return self.eraser(img), target


class ColorJitter(object):
    def __init__(self, brightness=0.4, contrast=0.4, saturation=0.4, hue=0.4):
        self.brightness = self._check_input(brightness, 'brightness')
        self.contrast = self._check_input(contrast, 'contrast')
        self.saturation = self._check_input(saturation, 'saturation')
        self.hue = self._check_input(hue, 'hue', center=0, bound=(-0.5, 0.5),
                                     clip_first_on_zero=False)

    def _check_input(self, value, name, center=1, bound=(0, float('inf')), clip_first_on_zero=True):
        if isinstance(value, numbers.Number):
            if value < 0:
                raise ValueError("If {} is a single number, it must be non negative.".format(name))
            value = [center - float(value), center + float(value)]
            if clip_first_on_zero:
                value[0] = max(value[0], 0.0)
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            if not bound[0] <= value[0] <= value[1] <= bound[1]:
                raise ValueError("{} values should be between {}".format(name, bound))
        else:
            raise TypeError("{} should be a single number or a list/tuple with lenght 2.".format(name))

        # if value is 0 or (1., 1.) for brightness/contrast/saturation
        # or (0., 0.) for hue, do nothing
        if value[0] == value[1] == center:
            value = None
        return value

    def __call__(self, img, target):
        fn_idx = torch.randperm(4)
        for fn_id in fn_idx:
            if fn_id == 0 and self.brightness is not None:
                brightness = self.brightness
                brightness_factor = torch.tensor(1.0).uniform_(brightness[0], brightness[1]).item()
                img = F.adjust_brightness(img, brightness_factor)

            if fn_id == 1 and self.contrast is not None:
                contrast = self.contrast
                contrast_factor = torch.tensor(1.0).uniform_(contrast[0], contrast[1]).item()
                img = F.adjust_contrast(img, contrast_factor)

            if fn_id == 2 and self.saturation is not None:
                saturation = self.saturation
                saturation_factor = torch.tensor(1.0).uniform_(saturation[0], saturation[1]).item()
                img = F.adjust_saturation(img, saturation_factor)

            if fn_id == 3 and self.hue is not None:
                hue = self.hue
                hue_factor = torch.tensor(1.0).uniform_(hue[0], hue[1]).item()
                img = F.adjust_hue(img, hue_factor)

        return img, target


class Normalize(object):
    def __init__(self, mean, std, normalize_lines=True):
        self.mean = mean
        self.std = std
        self.normalize_lines = normalize_lines

    def __call__(self, image, target=None):
        image = F.normalize(image, mean=self.mean, std=self.std)
        if target is None:
            return image, None
        target = target.copy()
        h, w = image.shape[-2:]

        if "lines" in target:
            lines = target["lines"]
            if self.normalize_lines:
                lines = lines / torch.tensor([w, h, w, h], dtype=torch.float32)
            swap_mask = _endpoint_swap_mask(lines)
            lines, lines3d = _swap_line_endpoints(lines, target.get("lines3d"), swap_mask)
            target["lines"] = lines
            if lines3d is not None:
                target["lines3d"] = lines3d

        return image, target


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string
