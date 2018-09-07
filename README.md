# Deepmedic and Unet models for Brats2018 competition

We created two popular deep learning models DeepMedic and 3D U-Net in PyTorch for the purpose of brain tumor segmentation.

For more details about our methodology, please refer to our [paper](https://www.researchgate.net/publication/326549702_Brain_Tumor_Segmentation_and_Tractographic_Feature_Extraction_from_Structural_MR_Images_for_Overall_Survival_Prediction)

## Citation

The system was employed for our research presented in [1], where the we integrate multiple DeepMedics and 3D U-Nets in order to get a robust tumor segmentation mask. If the use of the software or the idea of the paper positively influences your endeavours, please cite [1].

[1] **Po-Yu Kao**, Thuyen Ngo, Angela Zhang, Jefferson Chen, and B. S. Manjunath, "[Brain Tumor Segmentation and Tractographic Feature Extraction from Structural MR Images for Overall Survival Prediction.](https://arxiv.org/abs/1807.07716)" arXiv preprint arXiv:1807.07716, 2018.


## Dependencies

Python3.6

Pytorch0.4

Install custom pytorch kernels from https://github.com/thuyen/multicrop

## How to run
Change:
```
experiments/settings.yaml
```
to point to data directories. These are general settings, applied to all
experiments. Additional experiment-specific configuration will overwrite
these.

Split data to 5 fold train/valid splits:
```
python split.py
```

Preprocess data (look at the script for more details):
```
python prep.py
```

Prepare parcellation data:
```
python data/parcellation.py
```

For `deepmedic` run:
```
python train.py --gpu 0 --cfg deepmedic_ce
```

For `Unet` run:
```
python train_unet.py --gpu 0 --cfg unet_dice2
```

To make predictions, run `predict.py` or `predict_unet.py` with similar arguments

To make submissions, look at `make_submission.py`
