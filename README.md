# Deepmedic and 3D U-Net models for [Brats2018 tumor segmentation competition](https://www.med.upenn.edu/sbia/brats2018/tasks.html)

We created two popular deep learning models DeepMedic and 3D U-Net in PyTorch for the purpose of brain tumor segmentation.

For more details about our methodology, please refer to our [paper](https://www.researchgate.net/publication/326549702_Brain_Tumor_Segmentation_and_Tractographic_Feature_Extraction_from_Structural_MR_Images_for_Overall_Survival_Prediction)

The performance of our proposed ensemble on BraTS 2018 dataset is shown in the following table:

|Dataset|Dice(ET)|Dice(WT)|Dice(TC)|Sensitivity(ET)|Sensitivity(WT)|Sensitivity(TC)|Specificity(ET)|Specificity(WT)|Specificity(TC)|Hausdorff95(ET)|Hausdorff95(WT)|Hausdorff95(TC)|
|---|---|---|---|---|---|---|---|---|---|---|---|---|
|Training|0.73547|0.90208|0.81348|0.79451|0.89877|0.80269|0.9945|0.99145|0.99343|32.2726|5.4231|9.3586|
|Validation|0.7875|0.90473|0.81347|0.81113|0.921|0.79803|0.99803|0.99427|0.99827|3.81183|4.3231|7.5531|

For the training set, we use 5-folds cross validation. ET:Enhancing tumor, WT: whole tumor, TC: tumor core.

## Citation

The system was employed for our research presented in [1], where the we integrate multiple DeepMedics and 3D U-Nets in order to get a robust tumor segmentation mask. We also utilize the brain parcellation masks for the purpose of bringing the location information to DeepMedic and 3D U-Net. If the use of the software or the idea of the paper positively influences your endeavours, please cite [1].

[1] **Po-Yu Kao**, Thuyen Ngo, Angela Zhang, Jefferson Chen, and B. S. Manjunath, "[Brain Tumor Segmentation and Tractographic Feature Extraction from Structural MR Images for Overall Survival Prediction.](https://arxiv.org/abs/1807.07716)" arXiv preprint arXiv:1807.07716, 2018.


## Dependencies

Python3.6

Pytorch0.4

Install custom pytorch kernels from https://github.com/thuyen/multicrop

## Required Python libraries

nibable, nipype, natsort, SimpleITK

`pip install nibable,nipype,natsort,SimpleITK`

## Required Sofware

FSL (https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FslInstallation)

## Existing Brain Parcellation in Subject Space for BraTS2018 dataset

The Harvard-Oxford subcortical atlases in subject space are stored at `BrainParcellation/HarvardOxford-sub`

## Create the HarvardOxford Subcortical Brain Parcellation to Subject Space for New BraTS Subject

```
python createBrainParcellation -i /DIR_TO_MR_T1.nii.gz -o /DIR_TO_SAVE_THE_BRAIN_PARCELLATION -n SUBJECT_NAME
```

The output brain parcellation will be named as SUBJECT_NAME_HarvardOxford-sub.nii.gz

## Using Brain Parcellation on DeepMedic and 3D U-Net

For using the brain parcellation on DeepMedic and 3D U-Net, please change the paths in `data/parcellation.py` accordingly

## How to run

### Change:

```
experiments/settings.yaml
```

to point to data directories. These are general settings, applied to all
experiments. Additional experiment-specific configuration will overwrite
these.

### Split data to 5 fold train/valid splits:

```
python split.py
```

### Preprocess data (look at the script for more details):

```
python prep.py
```

### Prepare parcellation data:

For using the brain parcellation for DeepMedic and 3D U-Net, please change the paths in `data/parcellation.py` accordingly

```
python data/parcellation.py
```

### For standard DeepMedic, run:
```
python train.py --gpu 0 --cfg deepmedic_ce
```

### For DeepMedic with 12-by-12-by-12 output mask, run: 
```
python train_12.py --gpu 0 --cfg deepmedic_ce_28x20x12
```

### For DeepMedic with 6-by-6-by-6 output mask, run: 
```
python train_6.py --gpu 0 --cfg deepmedic_ce_22X18X6
```

### For 3D U-Net run:
```
python train_unet.py --gpu 0 --cfg unet_dice2
```

### Prediction

To make predictions, run `predict.py`, `predict_6.py`, `predict_12.py` or `predict_unet.py` with similar arguments

### Submission

To make submissions, look at `make_submission.py`

### Memory Issue (Data Compression)

If you do not have enough memory to save the output probability maps, you are able to compress these maps to uint16 format by `compress_data.py`

To make submissions with compressed probability maps, please refer to `ensemble_methods.py`



Special thanks to [Thuyen Ngo](https://github.com/thuyen).