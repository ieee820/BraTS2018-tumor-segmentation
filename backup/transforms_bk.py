import math
import random
import collections
import numpy as np
import torch
from scipy import ndimage

from .rand import Constant, Uniform, Gaussian

class Base(object):
    def sample(self, h, w, t):
        return h, w, t

    def tf(self, img, k=0):
        return img

    def __call__(self, img, reuse=False):
        # image: nhwtc
        if not reuse:
            im = img if isinstance(img, np.ndarray) else img[0]
            h, w, t = im.shape[1:4]
            self.sample(h, w, t)

        if isinstance(img, collections.Sequence):
            return [self.tf(x, k) for k, x in enumerate(img)]

        return self.tf(img)

    def __str__(self):
        return 'Itendity()'


Identity = Base

from .geo import Spatial


# gemetric transformations, need a buffers
class Rot90(Base):
    def __init__(self, axes=(0, 1)):
        self.axes = axes

    def sample(self, h, w, t):
        #nhwct
        if self.axes == (1, 2):
            h, w = w, h

        elif self.axes == (1, 3):
            h, t = t, h

        elif self.axes == (2, 3):
            w, t = t, w

        return h, w, t

    def tf(self, img, k=0):
        return np.rot90(img, axes=self.axes)

    def __str__(self):
        return 'Rot90(axes=({}, {})'.format(*self.axes)



class HFlip(Base):
    def tf(self, img):
        #nhwtc
        img = img[:, ::-1]
        return img

    def __str__(self):
        return 'HFlip()'

class WFlip(Base):
    def tf(self, img, k=0):
        #nhwtc
        img = img[:, :, ::-1]
        return img

    def __str__(self):
        return 'WFlip()'

class TFlip(Base):
    def tf(self, img, k=0):
        #nhwtc
        img = img[:, :, :, ::-1]
        return img

    def __str__(self):
        return 'TFlip()'


class RandSelect(Base):
    def __init__(self, prob=0.5, tf=None):
        self.prob = prob
        self.ops  = tf if isinstance(tf, collections.Sequence) else (tf, )
        self.buff = False

    def sample(self, h, w, t):
        self.buff = random.random() < self.prob

        if self.buff:
            for op in self.ops:
                h, w, t = op.sample(h, w, t)

        return h, w, t

    def tf(self, img, k=0):
        if self.buff:
            for op in self.ops:
                img = op.tf(img)
        return img

    def __str__(self):
        if len(self.ops) == 1:
            ops = str(self.ops[0])
        else:
            ops = '[{}]'.format(', '.join([str(op) for op in self.ops]))
        return 'RandSelect({}, {})'.format(self.prob, ops)


class CenterCrop(Base):
    def __init__(self, size):
        self.size = size
        self.buffer = None

    def sample(self, h, w, t):
        size = self.size

        th, tw, tt = (h-size) // 2, (w-size) // 2, (t-size) // 2

        self.buffer = th, tw, tt

        return size, size, size


    def tf(self, img, k=0):
        #nhwtc
        th, tw, tt = self.buffer
        size = self.size
        return img[:, th:th+size, tw:tw+size, tt:tt+size]

    def __str__(self):
        return 'CenterCrop({})'.format(self.size)


class RandCrop(CenterCrop):
    def sample(self, h, w, t):
        size = self.size

        th = random.randint(0, h-size)
        tw = random.randint(0, w-size)
        tt = random.randint(0, t-size)

        self.buffer = th, tw, tt
        return size, size, size

    def __str__(self):
        return 'RandCrop({})'.format(self.size)


class Pad(Base):
    def __init__(self, pad):
        self.pad = pad
        self.px = tuple(zip([0]*len(pad), pad))

    def sample(self, h, w, t):

        h += self.pad[0]
        w += self.pad[1]
        t += self.pad[2]

        return h, w, t

    def tf(self, img, k=0):
        #nhwtc, nhwt
        dim = len(img.shape)
        return np.pad(img, self.px[:dim], mode='constant')

    def __str__(self):
        return 'Pad(({}, {}, {}))'.format(*self.pad)


## No buffers, color transformation
class Noise(Base):
    def __init__(self, sigma=0.1, channel=True, num=-1):
        self.sigma = sigma
        self.channel = channel
        self.num = num

    def tf(self, img, k=0):
        if self.num > 0 and k >= self.num:
            return img

        if self.channel:
            #nhwtc, hwtc, hwt
            shape = [1] if len(img.shape) < 4 else [img.shape[-1]]
        else:
            shape = img.shape
        return img * np.exp(self.sigma * torch.randn(shape).numpy())

    def __str__(self):
        return 'Noise()'


class GaussianBlur(Base):
    def __init__(self, sigma=Constant(1.5), app=-1):
        # 1.5 pixel
        self.sigma = sigma
        self.eps   = 0.001
        self.app = app

    def tf(self, img, k=0):
        if self.num > 0 and k >= self.num:
            return img

        # image is nhwtc
        for n in range(img.shape[0]):
            sig = self.sigma.sample()
            # sample each channel saperately to avoid correlations
            if sig > self.eps:
                if len(img.shape) < 5:
                    img[n] = ndimage.gaussian_filter(img[n], sig)
                else:
                    C = img.shape[-1]
                    for c in range(C):
                        img[n,..., c] = ndimage.gaussian_filter(img[n, ..., c], sig)
        return img

    def __str__(self):
        return 'GaussianBlur()'


class ToNumpy(Base):
    def __init__(self, num=-1):
        self.num = num

    def tf(self, img, k=0):
        if self.num > 0 and k >= self.num:
            return img
        return img.numpy()

    def __str__(self):
        return 'ToNumpy()'


class ToTensor(Base):
    def __init__(self, num=-1):
        self.num = num

    def tf(self, img, k=0):
        if self.num > 0 and k >= self.num:
            return img

        return torch.from_numpy(img)

    def __str__(self):
        return 'ToTensor'


class TensorType(Base):
    def __init__(self, types, num=-1):
        self.types = types # ('torch.float32', 'torch.int64')
        self.num = num

    def tf(self, img, k=0):
        if self.num > 0 and k >= self.num:
            return img
        # make this work with both Tensor and Numpy
        return img.type(self.types[k])

    def __str__(self):
        s = ', '.join([str(s) for s in self.types])
        return 'TensorType(({}))'.format(s)


class NumpyType(Base):
    def __init__(self, types, num=-1):
        self.types = types # ('float32', 'int64')
        self.num = num

    def tf(self, img, k=0):
        if self.num > 0 and k >= self.num:
            return img
        # make this work with both Tensor and Numpy
        return img.astype(self.types[k])

    def __str__(self):
        s = ', '.join([str(s) for s in self.types])
        return 'NumpyType(({}))'.format(s)


class Normalize(Base):
    def __init__(self, mean=0.0, std=1.0, num=-1):
        self.mean = mean
        self.std = std
        self.num = num

    def tf(self, img, k=0):
        if self.num > 0 and k >= self.num:
            return img
        img -= self.mean
        img /= self.std
        return img

    def __str__(self):
        return 'Normalize()'


class Compose(Base):
    def __init__(self, ops):
        if not isinstance(ops, collections.Sequence):
            ops = ops,
        self.ops = ops

    def sample(self, h, w, t):
        for op in self.ops:
            h, w, t = op.sample(h, w, t)

    def tf(self, img, k=0):
        #is_tensor = isinstance(img, torch.Tensor)
        #if is_tensor:
        #    img = img.numpy()

        for op in self.ops:
            img = op.tf(img, k) # do not use op(img) here

        #if is_tensor:
        #    img = np.ascontiguousarray(img)
        #    img = torch.from_numpy(img)

        return img

    def __str__(self):
        ops = ', '.join([str(op) for op in self.ops])
        return 'Compose([{}])'.format(ops)

