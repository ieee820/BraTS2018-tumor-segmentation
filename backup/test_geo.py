#from transforms import Compose, Rot90, HFlip, VFlip, Affine, CenterCrop, Identity
#from transforms import RandSelect, RandCrop, RandSizedCrop
#from transforms import Noise, GaussianBlur

from geo import Spatial
from rand import Constant, Uniform, Gaussian


#crop = RandomCrop()
#while not done:
#    m = crop(mask)
#    done, uniques = is_valid(m) # check sum
#img = crop(img, reuse=True)
#img, m = transforms([img, m])



import numpy as np
import random

random.seed(100)
import cv2
im = cv2.imread('cat.jpg')
h, w = im.shape[:2]
im = im[None]

patch_size = 128, 128
transform = Spatial(
        patch_size,
        alpha=Uniform(0, 1000), sigma=Uniform(10, 13),
        angle_x=Uniform(0, 2*np.pi),
        scale=Uniform(0.85, 1.15))
transform.sample(h, w)
x = im[..., -1]
print(x.shape)
x = transform.tf(x)
print(im.shape, x.shape)
exit(0)

cv2.imwrite('out1.jpg', out1)
cv2.imwrite('out2.jpg', out2)
cv2.imwrite('out3.jpg', out3)


