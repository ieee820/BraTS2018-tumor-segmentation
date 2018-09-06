import os
import nibabel as nib
import numpy as np


settings = {
        'ensemble': {
            'models': ['deepmedic_ce_50_50_c25_all', 'deepmedic_ce_50_50_c25_all_noaug', 'unet_dice_all', 'deepmedic_ce_all', 'unet_ce_hard_per_im', 'unet_ce_hard', 'deepmedic_ce_60_80_100_b50_mb50_all', 'deepmedic_ce_90_120_150_b50_mb50_all'],
            'weights': [0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125],
            'note': 'integrate all models',
            },
        'ensemble_9': {
            'models': ['deepmedic_ce_50_50_c25_all', 'deepmedic_ce_50_50_c25_all_noaug', 'unet_dice_all', 'deepmedic_ce_all', 'unet_ce_hard_per_im', 'unet_ce_hard', 'deepmedic_ce_60_80_100_b50_mb50_all', 'deepmedic_ce_90_120_150_b50_mb50_all', 'deepmedic_ce_c25_60_80_100_b50_mb50_all'],
            'weights': [1, 1, 1, 1, 1, 1, 1, 1, 1],
            'note': 'integrate all 9 models',
            },
        'ensemble_26': {
            'models': [
            'deepmedic_ce_50_50_c25_all',
            'deepmedic_ce_50_50_c25_all_noaug', 
            'unet_dice_all', 
            'deepmedic_ce_all', 
            'unet_ce_hard_per_im', 
            'unet_ce_hard', 
            'deepmedic_ce_60_80_100_b50_mb50_all', 
            'deepmedic_ce_90_120_150_b50_mb50_all',
            'deepmedic_ce_c25_60_80_100_b50_mb50_all',
            'deepmedic_ce_c25_90_120_150_b50_mb50_all',
            'deepmedic_ce_c25_45_60_75_b50_mb50_all',
			'deepmedic_ce_c25_75_100_125_b50_mb50_all', 
			'deepmedic_ce_all_aug',
			'deepmedic_ce_50_50_all', 
			'deepmedic_ce_50_50_all_aug', 
			'deepmedic_ce_22x18x6_all_aug', 
			'deepmedic_ce_28x20x12_all_aug', 
			'deepmedic_ce_60_80_100_b50_mb50_all_aug', 
			'deepmedic_ce_90_120_150_b50_mb50_all_aug', 
			'deepmedic_ce_75_100_125_b50_mb50_all_aug', 
			'deepmedic_ce_75_100_125_b50_mb50_all',
            'deepmedic_ce_45_60_75_b50_mb50_all',
            'munet_dice_all', 
            'unet_dice_c25_all',
            'unet_ce_hard_c25',
            'unet_ce_hard_per_im_c25',
             ],
            'weights': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            'note': 'integrate all 26 models',
            },
        'deepmedic_c25': { #1
            'models': ['deepmedic_ce_50_50_c25_all'],
            'weights': [1.0],
            'note': 'deepmedic with all training datasets and Harvard Oxford Subcortical Atlas',
            },
        'deepmedic_c25_noaug': { #2
            'models': ['deepmedic_ce_50_50_c25_all_noaug'],
            'weights': [1.0],
            'note': 'deepmedic with all training datasets and Harvard Oxford Subcortical Atlas without data augmentation',
            },
        'unet': { #3
            'models': ['unet_dice_all'],
            'weights': [1.0],
            'note': 'unet',
            },
        'deepmedic_ce_all': { #4
            'models' :['deepmedic_ce_all'],
            'weights':[1.0],
            'note': 'deepmedic with all training datasets',
            },
        'unet_ce_hard_per_im': { #5
            'models': ['unet_ce_hard_per_im'],
            'weights': [1.0],
            'note': 'unet cross entropy loss hard mining per image',
            },
        'unet_ce_hard': { #6
            'models': ['unet_ce_hard'],
            'weights': [1.0],
            'note': 'unet cross entropy loss hard mining',
            }, 

        'deepmedic_double': { #7
            'models': ['deepmedic_ce_60_80_100_b50_mb50_all'],
            'weights': [1.0],
            'note': 'deepmedic with double convolutional kernels',  
            },
        'deepmedic_triple': { #8
            'models': ['deepmedic_ce_90_120_150_b50_mb50_all'],
            'weights': [1.0],
            'note': 'deepmedic with triple convolutional kernels',  
            } ,
        'deepmedic_double_c25': { #9
            'models': ['deepmedic_ce_c25_60_80_100_b50_mb50_all'],
            'weights': [1.0],
            'note': 'deepmedic with double convolutional kernels and 21 brain parcellatio channels',  
            } ,
        'deepmedic_triple_c25_aug': { #10
            'models': ['deepmedic_ce_c25_90_120_150_b50_mb50_all'],
            'weights': [1.0],
            'note': 'deepMedic Triple with 21 BPs and data augmentation',  
            } ,
        'deepmedic_15_c25_aug': { #11
            'models': ['deepmedic_ce_c25_45_60_75_b50_mb50_all'],
            'weights': [1.0],
            'note': 'Deepmedic with 1.5 kernels 21 BPs and data augment',  
            } ,
        'deepmedic_25_c25_aug': { #12
            'models': ['deepmedic_ce_c25_75_100_125_b50_mb50_all'],
            'weights': [1.0],
            'note': 'Deepmedic with 2.5 kernels 21 BPs and data augment',  
            } ,
        'deepmedic_ce_all_aug': { #13
            'models': ['deepmedic_ce_all_aug'],
            'weights': [1.0],
            'note': 'Deepmedic with data augment',  
            } ,
        'deepmedic_ce_50_50_all': { #14
            'models': ['deepmedic_ce_50_50_all'],
            'weights': [1.0],
            'note': 'Deepmedic mb50 b 50 without data augment',  
            } ,
        'deepmedic_ce_50_50_all_aug': { #15
            'models': ['deepmedic_ce_50_50_all_aug'],
            'weights': [1.0],
            'note': 'Deepmedic mb50 b 50 with data augment',  
            } ,
        'deepmedic_ce_22x18x6_all_aug': { #16
            'models': ['deepmedic_ce_22x18x6_all_aug'],
            'weights': [1.0],
            'note': 'deepmedic ce with 22x18x6 and data augment',  
            } ,
        'deepmedic_ce_28x20x12_all_aug': { #17
            'models': ['deepmedic_ce_28x20x12_all_aug'],
            'weights': [1.0],
            'note': 'deepmedic ce with 28x20x12 and data augment',  
            } ,
        'deepmedic_ce_60_80_100_b50_mb50_all_aug': { #18
            'models': ['deepmedic_ce_60_80_100_b50_mb50_all_aug'],
            'weights': [1.0],
            'note': 'deepmedic double with data aug',  
            } ,
        'deepmedic_ce_90_120_150_b50_mb50_all_aug': { #19
            'models': ['deepmedic_ce_90_120_150_b50_mb50_all_aug'],
            'weights': [1.0],
            'note': 'deepmedic triple with data aug',  
            } ,
        'deepmedic_ce_75_100_125_b50_mb50_all_aug': { #20
            'models': ['deepmedic_ce_75_100_125_b50_mb50_all_aug'],
            'weights': [1.0],
            'note': 'deepmedic 2.5 with data aug',  
            } ,
        'deepmedic_ce_75_100_125_b50_mb50_all': { #21
            'models': ['deepmedic_ce_75_100_125_b50_mb50_all'],
            'weights': [1.0],
            'note': 'deepmedic 2.5 without data aug',  
            } ,
        'deepmedic_ce_45_60_75_b50_mb50_all': { #22
            'models': ['deepmedic_ce_45_60_75_b50_mb50_all'],
            'weights': [1.0],
            'note': 'deepmedic 1.5 without data aug',  
            } ,
        'munet_dice_all': { #23
            'models': ['munet_dice_all'],
            'weights': [1.0],
            'note': 'modified 3D Unet with Dice loss',  
            } ,
        'unet_dice_c25_all': { #24
            'models': ['unet_dice_c25_all'],
            'weights': [1.0],
            'note': 'Unet with Dice Loss and BPs',  
            } ,
        'unet_ce_hard_c25': { #25
            'models': ['unet_ce_hard_c25'],
            'weights': [1.0],
            'note': 'Unet with ce and BPs',  
            } ,
        'unet_ce_hard_per_im_c25': { #26
            'models': ['unet_ce_hard_per_im_c25'],
            'weights': [1.0],
            'note': 'Unet with ce and BPs',  
            } ,
        }


root = '/usr/data/pkao/brats2018/validation'
file_list = os.path.join(root, 'test.txt')
names = open(file_list).read().splitlines()

root = './output'

#submission_name = 'deepmedic'
#submission_name = 'deepmedic_unet'
#submission_name = 'deepmedic_c25_noaug'
#submission_name = 'deepmedic_ce_all'
#submission_name = 'unet'
#submission_name = 'unet_ce_hard_per_im'
#submission_name= 'unet_ce_hard'
#submission_name='deepmedic_double'
#submission_name='deepmedic_triple'
#submission_name='deepmedic_double_c25'
#submission_name= 'ensemble_9'
#submission_name = 'deepmedic_triple_c25_aug'
#submission_name = 'deepmedic_15_c25_aug'
#submission_name = 'deepmedic_25_c25_aug'
#submission_name ='deepmedic_ce_all_aug'
#submission_name = 'deepmedic_ce_50_50_all'
#submission_name = 'deepmedic_ce_50_50_all_aug'
#submission_name = 'deepmedic_ce_22x18x6_all_aug'
#submission_name = 'deepmedic_ce_28x20x12_all_aug'
#submission_name = 'deepmedic_ce_60_80_100_b50_mb50_all_aug'
#submission_name = 'deepmedic_ce_90_120_150_b50_mb50_all_aug'
#submission_name ='deepmedic_ce_75_100_125_b50_mb50_all_aug'
#submission_name ='deepmedic_ce_75_100_125_b50_mb50_all'
#submission_name = 'ensemble_21'
#submission_name = 'deepmedic_ce_45_60_75_b50_mb50_all'
#submission_name = 'ensemble_22'
#submission_name = 'munet_dice_all'
#submission_name = 'ensemble_23_geo'
#submission_name = 'unet_dice_c25_all'
#submission_name = 'unet_ce_hard_c25'
#submission_name = 'unet_ce_hard_per_im_c25'
submission_name = 'ensemble_26'

models  = settings[submission_name]['models']
weights = settings[submission_name]['weights'] or [1.0] * len(models)

submission_dir = os.path.join('submissions', submission_name + '_uint8')
if not os.path.exists(submission_dir):
    os.makedirs(submission_dir)


for name in names:
    oname = os.path.join(submission_dir, name + '.nii.gz')
    preds = 0
    for k, model in enumerate(models):
        fname = os.path.join(root, models[k], 'test', name + '_preds.npy')
        
        # geometric mean
        #preds += weights[k] * np.log(np.load(fname)+0.001)
        # arithmetic mean
        preds += weights[k] * np.load(fname)
        ##preds += weights[k] * (255*np.load(fname)).astype('uint8')

    preds = preds.argmax(0).astype('uint8')

    img = nib.Nifti1Image(preds, None)
    nib.save(img, oname)
