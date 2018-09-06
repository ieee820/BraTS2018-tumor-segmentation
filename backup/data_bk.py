import sys
if sys.version_info[0] == 2:
    import Queue as queue
else:
    import queue
import os
import math
import multiprocessing as mp
import threading
import torch
from torch.utils.data import Dataset


import numpy as np
from data_utils import get_receptive_field, get_sub_patch_shape, \
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

        samples = samples - mean
        samples = samples / std

        sub_samples = sub_samples - mean
        sub_samples = sub_samples/std

        return samples, sub_samples, labels


    def __call__(self, index):
        return self.__getitem__(index)

    def __getitem__(self, index):
        path = self.names[index]

        start = time.time()
        images = np.array([
            nib_load(path + modal + '.nii.gz') \
            for modal in self.modalities])
        t1 = time.time() - start
        start = time.time()

        mask = images.sum(0) > 0

        #images -= mean * mask
        #images /= std

        label_file = path + 'seg.nii.gz'
        label = nib_load(label_file)
        exit(0)

        n = self.sample_size
        if self.split == 'train':
            fg = (label > 0).astype('int32')
            bg = ((mask > 0) * (fg == 0)).astype('int32')
            coords = np.concatenate(
                    [sample_coords(n/2, self.patch_shape, weight) for weight in (fg, bg)])
        elif self.split == 'valid':
            coords = sample_coords(n, self.patch_shape, mask)
        else: # test
            coords = get_all_coords((9, 9, 9), self.patch_shape, SHAPE, 15)
        t2 = time.time() - start
        start = time.time()
        samples, sub_samples, labels = self.crop(coords, images, label)
        t3 = time.time() - start

        msg = 'read {}, sample {}, crop {}, total {}'.format(t1, t2, t3, t1 + t2 + t3)
        print(msg)
        #print(t1, t2, t3, t1+t2+t3)
        # 2.3 sec total
        #exit(0)

        return samples, sub_samples, labels, coords


    def __len__(self):
        return len(self.names)

    @staticmethod
    def collate(batch):
        data = [torch.cat([torch.from_numpy(t) for t in v]) for v in zip(*batch)]
        perm = torch.randperm(data[0].shape[0])
        return [t[perm] for t in data]

class PEDataLoader(object):
    """
    A multiprocess-dataloader that parallels over elements as suppose to
    over batches (the torch built-in one)
    Input dataset must be callable with index argument: dataset(index)
    https://github.com/thuyen/nnet/blob/master/pedataloader.py
    """

    def __init__(self, dataset, batch_size=1, shuffle=False,
                 num_workers=None, pin_memory=False, num_batches=None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers

        self.collate_fn = collate
        self.pin_memory_fn = \
                torch.utils.data.dataloader.pin_memory_batch if pin_memory else \
                lambda x: x

        self.num_samples = len(dataset)
        self.num_batches = num_batches or \
                int(math.ceil(self.num_samples / float(self.batch_size)))

        self.pool = mp.Pool(num_workers)
        self.buffer = queue.Queue(maxsize=1)
        self.start()

    def generate_batches(self):
        if self.shuffle:
            indices = torch.LongTensor(self.batch_size)
            for b in range(self.num_batches):
                indices.random_(0, self.num_samples-1)
                batch = self.pool.map(self.dataset, indices)
                batch = self.collate_fn(batch)
                batch = self.pin_memory_fn(batch)
                yield batch
        else:
            self.indices = torch.LongTensor(range(self.num_samples))
            for b in range(self.num_batches):
                start_index = b*self.batch_size
                end_index = (b+1)*self.batch_size if b < self.num_batches - 1 \
                        else self.num_samples
                indices = self.indices[start_index:end_index]
                batch = self.pool.map(self.dataset, indices)
                batch = self.collate_fn(batch)
                batch = self.pin_memory_fn(batch)
                yield batch


    def start(self):
        def _thread():
            for b in self.generate_batches():
                self.buffer.put(b, block=True)
            self.buffer.put(None)

        thread = threading.Thread(target=_thread)
        thread.daemon = True
        thread.start()

    def __next__(self):
        batch = self.buffer.get()
        if batch is None:
            self.start()
            raise StopIteration
        return batch

    next = __next__

    def __iter__(self):
        return self

    def __len__(self):
        return self.num_batches

root = '/home/thuyen/Data/brats17/Brats17TrainingData/'
file_list = root + 'file_list.txt'
dset = ImageList(file_list, root=root)
import time
start = time.time()
for i in range(len(dset)):
    dset[i]
    #x1, x2, y, c = dset[0]
    print(time.time() - start)
    start = time.time()

exit(0)

from dataloader import DataLoader
import time

from sampler import SSampler


batch_size = 10
num_epochs = 20
num_iters = len(dset) * num_epochs // batch_size

sampler = SSampler(len(dset), num_epochs=num_epochs)

dloader = DataLoader(dset,
        batch_size=batch_size, pin_memory=True, collate_fn=ImageList.collate, sampler=sampler,
        #batch_size=batch_size, pin_memory=True, shuffle=True,
        #num_batches = num_iters,
        num_workers=20)

import torch
#a = torch.rand(10).cuda()

start = time.time()
count = 0
for k, x in enumerate(dloader):
    if k == 0:
        count = 0
        start = time.time()
    shapes = [t.shape for t in x]
    print(k, str(shapes))
    y = [t.cuda(non_blocking=True) for t in x]
    count += 1
end = time.time()
print((end-start)/(count - 1), count)


#start = time.time()
#for x in dloader:
#    end = time.time()
#    print(x[0].size(), end-start)
#    start = end
#
#exit(0)

# preprocess data to speedup traning and predictions
