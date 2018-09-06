import os
import random
from data.datasets import DualData, DualData1

train_list = 'train_0.txt'
root = '/home/thuyen/Data/brats17/Brats17TrainingData'
train_list = os.path.join(root, train_list)
dset0 = DualData(train_list, root=root, for_train=True)  # torch.from_numpy
dset1 = DualData1(train_list, root=root, for_train=True) # torch.tensor

for k in range(10):
    random.seed(k + 1000)
    y0 = dset0[k][0][-1]
    random.seed(k + 1000)
    y1 = dset1[k][0][-1]

    #print(y0.max().item(), y1.max().item())
    #print(y0.shape, y1.shape)
    print(k, (y0 == y1).all().item())
