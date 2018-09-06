import pickle
import os
import numpy as np
import nibabel as nib

suffix = '25x25x25_'
patch_shape = (25, 25, 25)

ndim = len(patch_shape)
dist2center = np.zeros((ndim, 2) , dtype='int32') # from patch boundaries
for dim, shape in enumerate(patch_shape) :
    dist2center[dim] = [shape/2 - 1, shape/2] if shape % 2 == 0 \
            else [shape//2, shape//2]

modalities = ('flair', 't1ce', 't1', 't2')
root = '/home/thuyen/Data/brats17/Brats17TrainingData/'
file_list = root + 'file_list.txt'
subjects = open(file_list).read().splitlines()

names = [sub.split('/')[-1] for sub in subjects]
paths = [os.path.join(root, sub, name + '_') for sub, name in zip(subjects, names)]

has_label = True

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

N = 50000*2
def process(path):
        label = 1 if not has_label else \
                nib_load(path + 'seg.nii.gz').astype('int16') # some are uint8

        uqs = np.unique(label)
        print(uqs)
        return

        images = np.stack([
            nib_load(path + modal + '.nii.gz') \
            for modal in modalities], -1)

        output = path + 'data.pkl'
        with open(output, 'wb') as f:
            pickle.dump((images, label), f)

        if not has_label:
            return

        mask = images.sum(-1) > 0
        sx, sy, sz = dist2center[:, 0]                # left-most boundary
        ex, ey, ez = mask.shape - dist2center[:, 1]   # right-most boundary
        shape = mask.shape
        maps = np.zeros(shape, dtype="int16")
        maps[sx:ex, sy:ey, sz:ez] = 1

        fg = (label > 0).astype('int16')
        bg = ((mask > 0) * (fg == 0)).astype('int16')

        fg = fg * maps
        bg = bg * maps

        fg = np.stack(fg.nonzero()).T.astype('int16')
        bg = np.stack(bg.nonzero()).T.astype('int16')

        fg = sample(fg, min(N, fg.shape[0]))
        bg = sample(bg, min(N, bg.shape[0]))

        output = path + suffix + 'coords.pkl'
        with open(output, 'wb') as f:
            pickle.dump((fg, bg), f)


for path in paths:
    print(path)
    process(path)
