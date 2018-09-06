import numpy as np
import torch
import multicrop

x = torch.randint(0, 5, (10, 15), dtype=torch.int16)

_stride = 5
_shape = (10, 15)


coords = torch.tensor(
        np.stack([v.reshape(-1) for v in
            np.meshgrid(
                *[_stride//2 + np.arange(0, s, _stride) for s in _shape], indexing='ij')], -1),
        dtype=torch.int16)


y = multicrop.crop2d(x, coords, 5, 5, 1, False)
#for t in y:
#    print('='*10)
#    print(t)
y = y.view(2, 3, 5, 5).permute(0, 2, 1, 3).reshape(10, 15)

print((x==y).all().item())

