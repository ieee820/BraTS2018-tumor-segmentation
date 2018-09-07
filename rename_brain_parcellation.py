import os
import shutil

brats2018_training_dir = '/home/pkao/BraTS2018-tumor-segmentation/BrainParcellation/HarvardOxford-sub'

brain_parcellation_old_name = 'HarvardOxford-sub-maxprob-thr0-1mm'

brain_parcellation_new_name = 'HarvardOxford-sub'

brain_parcellation_dirs = [os.path.join(root, name) for root, dirs, files in os.walk(brats2018_training_dir) for name in files 
						if brain_parcellation_old_name in name and name.endswith('.gz')]

print(len(brain_parcellation_dirs))

for old_name in brain_parcellation_dirs:
	
	new_name = old_name.replace(brain_parcellation_old_name, brain_parcellation_new_name)
	shutil.move(old_name, new_name)
