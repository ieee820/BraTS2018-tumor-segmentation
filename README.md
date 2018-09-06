# Deepmedic and Unet models for Brats2017 competition

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
