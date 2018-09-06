import numpy as np
import torch
import multicrop

x = torch.randint(0, 5, (10, 15, 10, 3), dtype=torch.int16)

_stride = 5
_shape = (10, 15, 10)


coords = torch.tensor(
        np.stack([v.reshape(-1) for v in
            np.meshgrid(
                *[_stride//2 + np.arange(0, s, _stride) for s in _shape], indexing='ij')], -1),
        dtype=torch.int16)


#x = x.cuda()
#coords = coords.cuda()
y = multicrop.crop3d_cpu(x, coords, 5, 5, 5, 1, False)
#for t in y:
#    print('='*10)
#    print(t)
y = y.view(2, 3, 2, 5, 5, 5, 3).permute(0, 3, 1, 4, 2, 5, 6).reshape(10, 15, 10, 3)

print((x==y).all().item())

