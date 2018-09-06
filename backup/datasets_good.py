#echo $OMP_NUM_THREADS
import sys
import pickle
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

import multicrop


#np.random.seed(2017)

#def sample_coords(size, weight):
#    indices = torch.multinomial(weight.view(-1), size, replacement=True)
#    v = np.unravel_index(indices, weight.shape)              # numpy take torch!
#    return torch.stack(
#            [torch.tensor(t, dtype=torch.int16) for t in v]) # but torch cannot

def sample(x, size):
    x = np.random.permutation(x)
    return torch.tensor(x[:size])


def pkload(fname):
    with open(fname, 'rb') as f:
        return pickle.load(f)

def normalize(x):
    return x/1000.0

eps = 1e-1
_stride = 9
_shape = (240, 240, 155)
_all_coords = torch.tensor(
        np.stack([v.reshape(-1) for v in
            np.meshgrid(
                *[_stride//2 + np.arange(0, s, _stride) for s in _shape], indexing='ij')], -1),
        dtype=torch.int16)

_zero = torch.tensor([0])

suffix = '25x25x25_'
sample_size, sub_sample_size, target_size = 25, 19, 9

class DualData(Dataset):
    def __init__(self, list_file, root='', sample_size=20, for_train=False,
            geo_transforms='', color_transforms='', return_target=True):
        names = []
        with open(list_file) as f:
            for line in f:
                line = line.strip()
                name = line.split('/')[-1]
                path = os.path.join(root, line , name + '_')
                names.append(path)

        self.root = root
        self.names = names
        self.sample_size = sample_size
        self.for_train = for_train

        self.return_target = return_target

        #self.geo_transforms   = eval(geo_transforms or 'Identity')
        #self.color_transforms = eval(color_transforms or 'Identity')

    def __getitem__(self, index):
        path = self.names[index]

        # faster than niffty
        images, label = pkload(path + 'data.pkl')
        images, label = torch.tensor(images), torch.tensor(label)

        if self.for_train:
            coords = pkload(path + suffix + 'coords.pkl')
            n = self.sample_size // 2
            coords  = torch.cat([sample(x, n) for x in coords])
        else:
            coords = _all_coords

        samples = multicrop.crop3d(images, coords,
                sample_size, sample_size, sample_size, 1, False).float()

        sub_samples = multicrop.crop3d(images, coords,
                sub_sample_size, sub_sample_size, sub_sample_size, 3, False).float()


        target = coords if not self.return_target else \
                multicrop.crop3d(
                        label, coords,
                        target_size, target_size, target_size, 1, False).long()

        if self.for_train: label = _zero


        ## data augmentation
        #samples, sub_samples, targets = self.geo_transforms(
        #        [samples, sub_samples, targets])

        #samples, sub_samples = self.color_transforms(
        #        [samples, sub_samples])

        samples = normalize(samples)
        sub_samples = normalize(sub_samples)


        samples = samples.permute(0, 4, 1, 2, 3).contiguous()
        sub_samples = sub_samples.permute(0, 4, 1, 2, 3).contiguous()

        return (samples, sub_samples, target), label


    def __len__(self):
        return len(self.names)

    def collate(self, batch):
        data, label = list(zip(*batch))
        data  = [torch.cat(v) for v in zip(*data)]
        label = torch.cat(label)

        if self.for_train:
            perm = torch.randperm(data[0].shape[0])
            data = [t[perm] for t in data]

            #counts = eps + torch.tensor(
            #        np.bincount(data[-1].view(-1).numpy(), minlength=5)).float()
            #weights = 1.0/5.0/counts * data[-1].numel()
            #data.append(weights)

        return data, label

#root = '/home/thuyen/Data/brats17/Brats17TrainingData/'
#file_list = root + 'file_list.txt'
##dset = DualData(file_list, root=root, for_train=True)
#dset = DualData(file_list, root=root, for_train=False)
#import time
#start = time.time()
##for i in range(len(dset)):
#for i in range(10):
#    dset[i]
#    #x1, x2, y, c = dset[0]
#    print(time.time() - start)
#    start = time.time()

