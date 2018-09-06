import torch
import numpy as np
import os
import random
import time
from data.datasets import DualData

train_list = 'train_0.txt'
root = '/home/thuyen/Data/brats17/Brats17TrainingData'
train_list = os.path.join(root, train_list)
dset = DualData(train_list, root=root, for_train=True)  # torch.from_numpy

start = time.time()
for k in range(10):
    y = dset[k][0][-1]

print(time.time() - start)

