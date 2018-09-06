import pickle
import hickle
import os
import numpy as np
import nibabel as nib
from utils import Parser

args = Parser()

suffix = '25x25x25_'
patch_shape = (25, 25, 25)


ndim = len(patch_shape)
dist2center = np.zeros((ndim, 2) , dtype='int32') # from patch boundaries
for dim, shape in enumerate(patch_shape) :
    dist2center[dim] = [shape/2 - 1, shape/2] if shape % 2 == 0 \
            else [shape//2, shape//2]

modalities = ('flair', 't1ce', 't1', 't2')

# train
root = args.data_dir
file_list = os.path.join(root, 'all.txt')
has_label = True
####

# test/validation data
root = '/home/thuyen/Data/brats17/Brats17ValidationData'
file_list = os.path.join(root, 'test.txt')
has_label = False
####


subjects = open(file_list).read().splitlines()

names = [sub.split('/')[-1] for sub in subjects]
paths = [os.path.join(root, sub, name + '_') for sub, name in zip(subjects, names)]


def sample(x, size):
    x = np.random.permutation(x)
    return x[:size]

def nib_load(file_name):
    proxy = nib.load(file_name)
    data = proxy.get_data()
    #print('thuyen', data.dtype)
    #data = data.astype('float32')
    proxy.uncache()
    return data

import time
N = 50000*2
def process(path, prefix=''):
    label = np.array([1], dtype='uint8') if not has_label else \
            nib_load(path + 'seg.nii.gz').astype('uint8') # some are uint8

    images = np.stack([
        nib_load(path + modal + '.nii.gz') \
        for modal in modalities], -1)

    #output = path + prefix + 'data.pkl'
    #with open(output, 'wb') as f:
    #    pickle.dump((images, label), f)

    #images = images.astype('float32')
    mask  = images.sum(-1) > 0
    mask = mask.astype('uint8')

    images = images.astype('float32')

    mean, std = [], []
    for k in range(4):
        x = images[..., k]
        y = x[mask > 0]
        lower = np.percentile(y, 0.2)
        upper = np.percentile(y, 99.8)
        x[(mask > 0) & (x < lower)] = lower
        x[(mask > 0) & (x > upper)] = upper

        y = x[mask > 0]

        x -= y.mean()
        x /= y.std()


        # cvt to uint16
        #x = np.clip(x, -10, 10)
        #x = x*3000 + 30000

        mean.append(x.min())
        std.append(x.max())

        images[..., k] = x


    #return mean + std

    #output = path + prefix + 'data_f32.pkl'
    output = 'data_f32.pkl'
    with open(output, 'wb') as f:
        pickle.dump((images, label), f)

    exit(0)


    return 0, 0


    if not has_label:
        return

    sx, sy, sz = dist2center[:, 0]                # left-most boundary
    ex, ey, ez = mask.shape - dist2center[:, 1]   # right-most boundary
    shape = mask.shape
    maps = np.zeros(shape, dtype="int16")
    maps[sx:ex, sy:ey, sz:ez] = 1

    fg = (label > 0).astype('int16')
    bg = ((mask > 0) * (fg == 0)).astype('int16')

    fg = fg * maps
    bg = bg * maps

    fg = np.stack(fg.nonzero()).T.astype('uint8')
    bg = np.stack(bg.nonzero()).T.astype('uint8')

    #fg = sample(fg, min(N, fg.shape[0]))
    #bg = sample(bg, min(N, bg.shape[0]))

    output = path + suffix + prefix + 'coords.pkl'
    with open(output, 'wb') as f:
        pickle.dump((fg, bg), f)


ret = []
for path in paths:
    print(path)
    ret.append(process(path))

ret = np.array(ret)
print(ret.min(0), ret.max(0))

#[ 91.751045  37.984158 190.77567   84.567215  28.206722  11.96204
#          41.119194  35.08886 ]
#[14568.448  12495.514  19225.385  13866.665   4324.063   2861.2239
#                    4857.4917  4535.893 ]
