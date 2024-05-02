import functools
import random
import math
from PIL import Image

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from datasets import register
from utils import to_pixel_samples


@register('sr-implicit-paired')
class SRImplicitPaired(Dataset):

    def __init__(self, dataset, inp_size=None, augment=False, sample_q=None):
        self.dataset = dataset
        self.inp_size = inp_size
        self.augment = augment
        self.sample_q = sample_q

    def update_container(self, name, container, cur_data):

        if name in container:
            container[name] = torch.cat((container[name], cur_data.unsqueeze(0)), dim=0)
        else:
            container[name] = cur_data.unsqueeze(0)
            
        return container
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        container = dict()
             
        img_lr, img_hr = self.dataset[idx]

        s = img_hr.shape[-2] // img_lr.shape[-2] # assume int scale
        if self.inp_size is None:
            h_lr, w_lr = img_lr.shape[-2:]
            img_hr = img_hr[:, :h_lr * s, :w_lr * s]
            crop_lr, crop_hr = img_lr, img_hr
        else:
            w_lr = self.inp_size
            x0 = random.randint(0, img_lr.shape[-2] - w_lr)
            y0 = random.randint(0, img_lr.shape[-1] - w_lr)
            crop_lr = img_lr[:, x0: x0 + w_lr, y0: y0 + w_lr]
            w_hr = w_lr * s
            x1 = x0 * s
            y1 = y0 * s
            crop_hr = img_hr[:, x1: x1 + w_hr, y1: y1 + w_hr]

        if self.augment:
            hflip = random.random() < 0.5
            vflip = random.random() < 0.5
            dflip = random.random() < 0.5

            def augment(x):
                if hflip:
                    x = x.flip(-2)
                if vflip:
                    x = x.flip(-1)
                if dflip:
                    x = x.transpose(-2, -1)
                return x

            crop_lr = augment(crop_lr)
            crop_hr = augment(crop_hr)


        hr_coord, hr_rgb = to_pixel_samples(crop_hr.contiguous())

        if self.sample_q is not None:
            sample_lst = np.random.choice(
                len(hr_coord), self.sample_q, replace=False)
            hr_coord = hr_coord[sample_lst]
            hr_rgb = hr_rgb[sample_lst]

        cell = torch.ones_like(hr_coord)
        cell[:, 0] *= 2 / crop_hr.shape[-2]
        cell[:, 1] *= 2 / crop_hr.shape[-1]

        container = self.update_container('inp', container, crop_lr)
        container = self.update_container('ref', container, crop_hr)
        container = self.update_container('coord', container, hr_coord)
        container = self.update_container('cell', container, cell)
        container = self.update_container('gt', container, hr_rgb)

        return container
	
def resize_fn(img, size):
    return transforms.ToTensor()(
        transforms.Resize(size, Image.BICUBIC)(
            transforms.ToPILImage()(img)))


@register('sr-implicit-downsampled')
class SRImplicitDownsampled(Dataset):

    def __init__(self, dataset, inp_size=None, scale_min=1, scale_max=None,
                 augment=False, sample_q=None, sample_patch=None, batch_size=1):
        self.dataset = dataset
        self.inp_size = inp_size
        self.scale_min = scale_min
        if scale_max is None:
            scale_max = scale_min
        self.scale_max = scale_max
        self.augment = augment
        self.sample_q = sample_q
        
        self.batch_size = batch_size # batch size for same scale
        self.sample_patch = sample_patch # samples a patch from an image
    
    def update_container(self, name, container, cur_data):
        if name in container:
            container[name] = torch.cat((container[name], cur_data.unsqueeze(0)), dim=0)
        else:
            container[name] = cur_data.unsqueeze(0)
            
        return container


    def __len__(self):
        return len(self.dataset) // self.batch_size

    def __getitem__(self, idx):
        container = dict()
        s = random.uniform(self.scale_min, self.scale_max)

        for i in range(self.batch_size):
            img = self.dataset[self.batch_size*idx+i]

            if self.inp_size is None:
                h_lr = math.floor(img.shape[-2] / s + 1e-9)
                w_lr = math.floor(img.shape[-1] / s + 1e-9)
                h_hr = round(h_lr * s)
                w_hr = round(w_lr * s)
                img = img[:, :h_hr, :w_hr] # assume round int
                img_down = resize_fn(img, (h_lr, w_lr))
                crop_lr, crop_hr = img_down, img
                
            else:
                h_lr = w_lr = self.inp_size
                h_hr = w_hr = round(w_lr * s)
                x0 = random.randint(0, img.shape[-2] - w_hr)
                y0 = random.randint(0, img.shape[-1] - w_hr)
                crop_hr = img[:, x0: x0 + w_hr, y0: y0 + w_hr]
                crop_lr = resize_fn(crop_hr, w_lr)
                

            if self.augment:
                hflip = random.random() < 0.5
                vflip = random.random() < 0.5
                dflip = random.random() < 0.5

                def augment(x):
                    if hflip:
                        x = x.flip(-2)
                    if vflip:
                        x = x.flip(-1)
                    if dflip:
                        x = x.transpose(-2, -1)
                    return x

                crop_lr = augment(crop_lr)
                crop_hr = augment(crop_hr)

            hr_coord, hr_rgb = to_pixel_samples(crop_hr.contiguous())

            if self.sample_patch is not None:
                h_idx = random.randint(0, h_hr-h_lr)
                w_idx = random.randint(0, w_hr-w_lr)
                pos = w_hr*h_idx + w_idx
                sample_lst = list()
                for i in range(h_lr):
                    start = pos + w_hr*i
                    end = start + w_lr
                    sample_lst += [j for j in range(start, end)]
                sample_lst = np.array(sample_lst)
                hr_coord = hr_coord[sample_lst]
                hr_rgb = hr_rgb[sample_lst]
            elif self.sample_q is not None: # samples random pixels
                sample_lst = np.random.choice(
                    len(hr_coord), self.sample_q, replace=False)
                hr_coord = hr_coord[sample_lst]
                hr_rgb = hr_rgb[sample_lst]

            cell = torch.ones_like(hr_coord)
            cell[:, 0] *= 2 / crop_hr.shape[-2]
            cell[:, 1] *= 2 / crop_hr.shape[-1]

            container = self.update_container('inp', container, crop_lr)
            container = self.update_container('ref', container, crop_hr)
            container = self.update_container('coord', container, hr_coord)
            container = self.update_container('cell', container, cell)
            container = self.update_container('gt', container, hr_rgb)
        
        return container

@register('sr-implicit-uniform-varied')
class SRImplicitUniformVaried(Dataset):

    def __init__(self, dataset, size_min, size_max=None,
                 augment=False, gt_resize=None, sample_q=None):
        self.dataset = dataset
        self.size_min = size_min
        if size_max is None:
            size_max = size_min
        self.size_max = size_max
        self.augment = augment
        self.gt_resize = gt_resize
        self.sample_q = sample_q

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img_lr, img_hr = self.dataset[idx]
        p = idx / (len(self.dataset) - 1)
        w_hr = round(self.size_min + (self.size_max - self.size_min) * p)
        img_hr = resize_fn(img_hr, w_hr)

        if self.augment:
            if random.random() < 0.5:
                img_lr = img_lr.flip(-1)
                img_hr = img_hr.flip(-1)

        if self.gt_resize is not None:
            img_hr = resize_fn(img_hr, self.gt_resize)

        hr_coord, hr_rgb = to_pixel_samples(img_hr)

        if self.sample_q is not None:
            sample_lst = np.random.choice(
                len(hr_coord), self.sample_q, replace=False)
            hr_coord = hr_coord[sample_lst]
            hr_rgb = hr_rgb[sample_lst]

        cell = torch.ones_like(hr_coord)
        cell[:, 0] *= 2 / img_hr.shape[-2]
        cell[:, 1] *= 2 / img_hr.shape[-1]

        return {
            'inp': img_lr,
            'coord': hr_coord,
            'cell': cell,
            'gt': hr_rgb,
        }
