import sys
if sys.version_info[0] == 2:
    import Queue as queue
else:
    import queue
import time
import os
import math
import multiprocessing as mp
import threading
import torch
from torch.utils.data import Dataset

import numpy as np
from .data_utils import get_receptive_field, get_sub_patch_shape, \
        get_offset, sample_coords, get_all_coords, nib_load

PATCH_SHAPE = (25, 25, 25)
KERNELS = ((3, 3, 3), )*8
SCALE_FACTOR = (3, 3, 3)
SHAPE = [240, 240, 155]

np.random.seed(2017)

mean = [433.78412444, 661.42844749, 588.09469198, 651.22305233]
mean = np.array(mean).reshape(4, 1, 1, 1)

std = [1343.81579289, 1200.61193295, 1178.99769383, 1390.22978543]
std = np.array(std).reshape(4, 1, 1, 1)

class ImageList(Dataset):
    def __init__(self,
            list_file,
            patch_shape=PATCH_SHAPE,
            kernels=KERNELS,
            scale_factor=SCALE_FACTOR,
            root='',
            split='valid',
            sample_size=20):

        names = []
        with open(list_file) as f:
            for line in f:
                line = line.strip()
                name = line.split('/')[-1]
                path = os.path.join(root, line , name + '_')
                names.append(path)

        self.root = root
        self.names = names
        self.split = split
        self.sample_size = sample_size
        self.receptive_field = get_receptive_field(kernels)
        self.patch_shape = np.array(patch_shape)
        self.scale_factor = np.array(scale_factor)
        self.sub_patch_shape = get_sub_patch_shape(self.patch_shape,
                self.receptive_field, self.scale_factor)
        self.sub_off = get_offset(self.scale_factor, self.receptive_field)
        self.modalities = ('flair', 't1ce', 't1', 't2')
        self.C = len(self.modalities)

    def coord_to_slice(self, coord):
        return coord[:, 0], coord[:, 1] + 1

    def coord_to_sub_slice(self, coord):
        lo = coord[:, 0] + self.sub_off
        num = self.patch_shape - self.receptive_field + 1
        hi = lo + self.scale_factor*self.receptive_field + \
                np.ceil((num*1.0)/self.scale_factor - 1) * self.scale_factor
        hi = hi.astype('int')
        lo = lo.astype('int')

        m = lo < 0
        pl = -lo * m
        lo[lo < 0] = 0

        m = hi > SHAPE
        ph = (hi - SHAPE) * m
        hi += pl.astype('int')

        pad = list(zip(pl, ph))
        return lo, hi, pad

    def crop(self, coords, images, label):
        N = coords.shape[0]
        samples = np.zeros((N, self.C) + tuple(self.patch_shape), dtype='float32')
        sub_samples = np.zeros((N, self.C) + tuple(self.sub_patch_shape), dtype='float32')
        labels = np.zeros((N,) + (9, 9, 9), dtype='int')

        size = (self.sub_patch_shape - 1)//2
        gl = (self.patch_shape - size)//2
        gh = self.patch_shape - gl

        kx, ky, kz = self.scale_factor


        for n, coord in enumerate(coords):
            ss, ee = self.coord_to_slice(coord)
            lo, hi, pad = self.coord_to_sub_slice(coord)

            cropped_label = label[ss[0]:ee[0], ss[1]:ee[1], ss[2]:ee[2]]
            labels[n] = cropped_label[gl[0]:gh[0], gl[1]:gh[1], gl[2]:gh[2]]

            samples[n] = images[:, ss[0]:ee[0], ss[1]:ee[1], ss[2]:ee[2]]

            pimages = np.pad(images, [(0, 0)] + pad, mode='constant') \
                    if np.sum(pad) > 0 else images

            sub_samples[n] = \
                    pimages[:, lo[0]:hi[0]:kx, lo[1]:hi[1]:ky, lo[2]:hi[2]:kz]

        return samples, sub_samples, labels


    def __call__(self, index):
        return self.__getitem__(index)

    def __getitem__(self, index):
        path = self.names[index]

        # sample and crops are expensive 0.6-0.8s

        # faster with int16, keep int16
        #start = time.time()
        images = np.array([
            nib_load(path + modal + '.nii.gz') \
            for modal in self.modalities])
        #t1 = time.time() - start
        #start = time.time()

        mask = images.sum(0) > 0
        # mask = xyz
        # imgs = cxyz # xyzc could be faster

        ## tod this later to save time, int16 as well
        #images -= mean * mask
        #images /= std

        # load uint8
        label_file = path + 'seg.nii.gz'
        label = nib_load(label_file)

        n = self.sample_size
        if self.split == 'train':
            # uint8 would be faster
            fg = (label > 0).astype('int32')
            bg = ((mask > 0) * (fg == 0)).astype('int32')
            coords = np.concatenate(
                    [sample_coords(n//2, self.patch_shape, weight) for weight in (fg, bg)])
        elif self.split == 'valid':
            coords = sample_coords(n, self.patch_shape, mask)
        else: # test
            coords = get_all_coords((9, 9, 9), self.patch_shape, SHAPE, 15)
        #t2 = time.time() - start
        #start = time.time()
        samples, sub_samples, labels = self.crop(coords, images, label)
        #t3 = time.time() - start
        #print(t1, t2, t3, t1+t2+t3)
        # 2.3 sec total
        #exit(0)

        sampels = samples.astype('float32')
        sampels -= mean
        sampels /= std

        sub_sampels = sub_samples.astype('float32')
        sub_sampels -= mean
        sub_sampels /= std


        return samples, sub_samples, labels, coords


    def __len__(self):
        return len(self.names)

    @staticmethod
    def collate(batch):
        data = [torch.cat([torch.from_numpy(t) for t in v]) for v in zip(*batch)]
        perm = torch.randperm(data[0].shape[0])
        return [t[perm] for t in data]

