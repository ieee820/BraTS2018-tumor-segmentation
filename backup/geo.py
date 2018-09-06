import numpy as np
from scipy.ndimage import zoom
from .utils import create_zero_centered_coordinate_mesh, elastic_deform_coordinates, \
    interpolate_img, \
    rotate_coords_2d, rotate_coords_3d, scale_coords

from .transforms import Base

#https://github.com/MIC-DKFZ/batchgenerators/blob/master/batchgenerators/augmentations/spatial_transformations.py#L25

#def augment_spatial(data, seg, patch_size, patch_center_dist_from_border=30,
#                    do_elastic_deform=True, alpha=(0., 1000.), sigma=(10., 13.),
#                    do_rotation=True, angle_x=(0, 2 * np.pi), angle_y=(0, 2 * np.pi), angle_z=(0, 2 * np.pi),
#                    do_scale=True, scale=(0.75, 1.25), border_mode_data='nearest', border_cval_data=0, order_data=3,
#                    border_mode_seg='constant', border_cval_seg=0, order_seg=0, random_crop=True):

#patch_size = 128, 128, 128
#dim = 3
## 3x128x128x128
#coords = create_zero_centered_coordinate_mesh(patch_size)

# gemetric transformations, need a buffers
class Spatial(Base):
    def __init__(self, patch_size, center_to_border=None,
            alpha=None, sigma=None,
            angle_x=None, angle_y=None, angle_z=None,
            scale=None, random_crop=True):

        self.patch_size = patch_size
        self.buff = None
        self.alpha = alpha
        self.sigma = sigma
        self.angle_x = angle_x
        self.angle_y = angle_y
        self.angle_z = angle_z
        self.scale = scale
        self.dim = len(patch_size)
        self.random_crop = random_crop

        if center_to_border is None:
            center_to_border = list(np.array(patch_size)//2)
        elif not isinstance(center_to_border, collections.Sequence):
            center_to_border = self.dim * (center_to_border, )
        else:
            raise ValueError('center to border')

        self.center_to_border = center_to_border



        self.order_data = 3
        self.border_mode_data = 'nearest'
        self.border_cval_data = 0

    def sample(self, *shape):
        #nhwtc
        coords = create_zero_centered_coordinate_mesh(self.patch_size)

        if self.alpha is not None and self.sigma is not None:
            a = self.alpha.sample()
            s = self.sigma.sample()
            coords = elastic_deform_coordinates(coords, a, s)

        if self.angle_x is not None:
            ax = self.angle_x.sample()
            if self.dim == 3:
                ay = self.angle_y.sample()
                az = self.angle_z.sample()
                coords = rotate_coords_3d(coords, ax, ay, az)
            else:
                coords = rotate_coords_2d(coords, ax)
        if self.scale is not None:
            sc = self.scale.sample()
            coords = scale_coords(coords, sc)

        for d in range(self.dim):
            if self.random_crop:
                ctr = np.random.uniform(self.center_to_border[d],
                                        shape[d] - self.center_to_border[d])
            else:
                ctr = int(np.round(shape[d] / 2.))
            coords[d] += ctr

        self.buff = coords

        return self.patch_size

    def tf(self, img):
        shape = list(img.shape)
        shape[1:self.dim+1] = self.patch_size
        out = np.zeros(shape, dtype=img.dtype)
        for n in range(img.shape[0]):
            if len(img.shape) == self.dim+2:
                for c in range(img.shape[-1]):
                    out[n, ..., c] = interpolate_img(img[n, ..., c], self.buff,
                            self.order_data, self.border_mode_data, cval=self.border_cval_data)
            elif len(img.shape) == self.dim+1:
                out[n] = interpolate_img(img[n], self.buff,
                        self.order_data, self.border_mode_data, cval=self.border_cval_data)
            else:
                raise ValueError('image shape is not supported')
        return out

