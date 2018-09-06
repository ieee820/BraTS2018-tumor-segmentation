import numpy as np
import nibabel as nib
import os

def nib_load(file_name):
    proxy = nib.load(file_name)
    data = proxy.get_data()
    proxy.uncache()
    return data

def nib_save(file_name, data):
    img = nib.Nifti1Image(data, np.eye(4))
    nib.save(img, file_name)

name = 'Brats17_TCIA_608_1'
root = '/home/thuyen/Data/brats17/Brats17TrainingData/'

file_list = root + 'file_list.txt'
modes = ['flair', 't1ce', 't1', 't2']

# 4x9x9x9    # n_sg.png
# 4x19x19x19 # n_x1.png, n_m1.png
# 4x25x25x25 # n_x2.png, n_m2.png

with open(file_list) as f:
    for line in f:
        line = line.strip()
        name = line.split('/')[-1]
        imgs = np.stack([nib_load(os.path.join(
            root, line , name + '_' + mode + '.nii.gz')) for mode in modes])
        path = os.path.join(
            root, line , name + '_mask.nii.gz')
        mask = nib_load(path)
        print(mask.shape)
        exit(0)
        #print(mask.sum())
        path = os.path.join(
            root, line , name + '_seg.nii.gz')
        segm = nib_load(path) > 0
        print(segm.sum(), mask.sum())

#with open(file_list) as f:
#    for line in f:
#        line = line.strip()
#        name = line.split('/')[-1]
#        imgs = np.stack([nib_load(os.path.join(
#            root, line , name + '_' + mode + '.nii.gz')) for mode in modes])
#        path = os.path.join(
#            root, line , name + '_mask.nii.gz')
#        mask = (imgs.sum(0) > 0).astype('uint8')
#        nib_save(path, mask)

#mean = 0.0
#count = 0.0
#with open(file_list) as f:
#    for line in f:
#        line = line.strip()
#        name = line.split('/')[-1]
#        imgs = np.stack([nib_load(os.path.join(
#            root, line , name + '_' + mode + '.nii.gz')) for mode in modes])
#        mask = imgs.sum(0) > 0
#        count += mask.sum()
#        mean += imgs.reshape(4, -1).sum(1)
#mean = mean/count
#print(mean)
#mean = [433.78412444, 661.42844749, 588.09469198, 651.22305233]
#
#mean = np.array(mean)
#mean = mean[:, None, None, None]
#
#var = 0.0
#count = 0.0
#with open(file_list) as f:
#    for line in f:
#        line = line.strip()
#        name = line.split('/')[-1]
#        imgs = np.stack([nib_load(os.path.join(
#            root, line , name + '_' + mode + '.nii.gz')) for mode in modes])
#        mask = imgs.sum(0, keepdims=True) > 0
#        count += mask.sum()
#        var += ((imgs - mean*mask)**2).reshape(4, -1).sum(1)
#var = var/count
#std = np.sqrt(var)
#print(std)
#
##std = [1343.81579289, 1200.61193295, 1178.99769383, 1390.22978543]
