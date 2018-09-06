import torch
import random

from torch.utils.data import Dataset, DataLoader

#class Data(Dataset):
#    def __len__(self):
#        return 10000
#    def __getitem__(self, index):
#        print(index, torch.rand(2, 2).sum().item(), random.uniform(0, 1))
#        return 1
#
#def init_fn(k):
#    random.seed(seed)
#
#
##seed = 2018
##random.seed(seed)
##torch.manual_seed(seed)
#torch.manual_seed(2018)
#loader = DataLoader(Data(), num_workers=4, shuffle=True, worker_init_fn=init_fn)
#
#for k, x in enumerate(loader):
#    print('-'*10)
#    break




class RandomDatasetMock(object):

    def __getitem__(self, index):
        return torch.tensor([torch.rand(1).item(), random.uniform(0, 1)])

    def __len__(self):
        return 1000

def run():
    dataloader = torch.utils.data.DataLoader(RandomDatasetMock(),
                                             num_workers=4,
                                             batch_size=2,
                                             shuffle=True)
    return next(iter(dataloader))

torch.manual_seed(2018)
x1 = run()
torch.manual_seed(2018)
x2 = run()


print(x1)
print(x2)
print(x1.dtype, x2.dtype)
print(x1.shape, x2.shape)
