import argparse
import os
import shutil
import time
import logging
import random

import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torch.optim
cudnn.benchmark = True

import multicrop
import numpy as np

from medpy import metric

import models
from models import criterions
from data import datasets
from data.data_utils import add_mask
from utils import Parser

path = os.path.dirname(__file__)

def calculate_metrics(pred, target):
    sens = metric.sensitivity(pred, target)
    spec = metric.specificity(pred, target)
    dice = metric.dc(pred, target)

eps = 1e-5
def f1_score(o, t):
    num = 2*(o*t).sum() + eps
    den = o.sum() + t.sum() + eps
    return num/den

#https://github.com/ellisdg/3DUnetCNN
#https://github.com/ellisdg/3DUnetCNN/blob/master/brats/evaluate.py
#https://github.com/MIC-DKFZ/BraTS2017/blob/master/utils_validation.py
def dice(output, target):
    ret = []
    # whole
    o = output > 0; t = target > 0
    ret += f1_score(o, t),
    # core
    o = (output==1) | (output==4)
    t = (target==1) | (target==4)
    ret += f1_score(o , t),
    # active
    o = (output==4); t = (target==4)
    ret += f1_score(o , t),

    return ret

keys = 'whole', 'core', 'enhancing', 'loss'
def main():
    ckpts = args.getdir()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    # setup networks
    Network = getattr(models, args.net)
    model = Network(**args.net_params)
    model = model.cuda()

    model_file = os.path.join(ckpts, args.ckpt)
    checkpoint = torch.load(model_file)
    model.load_state_dict(checkpoint['state_dict'])

    Dataset = getattr(datasets, args.dataset)

    valid_list = os.path.join(args.data_dir, args.valid_list)
    valid_set = Dataset(valid_list, root=args.data_dir,
            for_train=False, crop=False, return_target=args.scoring,
            transforms=args.test_transforms,
            sample_size=args.sample_size, sub_sample_size=args.sub_sample_size,
            target_size=args.target_size)
    valid_loader = DataLoader(
        valid_set, batch_size=1, shuffle=False,
        collate_fn=valid_set.collate,
        num_workers=4, pin_memory=True)

    start = time.time()
    with torch.no_grad():
        scores = validate(valid_loader, model, args.batch_size,
                args.out_dir, valid_set.names, scoring=args.scoring)

    msg = 'total time {:.4f} minutes'.format((time.time() - start)/60)
    logging.info(msg)


def validate(valid_loader, model, batch_size,
        out_dir='', names=None, scoring=True, verbose=True):

    H, W, T = 240, 240, 155

    dset = valid_loader.dataset
    names = dset.names
    h, w, t = dset.shape; h, w, t = int(h), int(w), int(t)
    sample_size = dset.sample_size
    sub_sample_size = dset.sub_sample_size
    target_size = dset.target_size
    dtype = torch.float32

    model.eval()
    criterion = F.cross_entropy

    vals = AverageMeter()
    for i, (data, labels) in enumerate(valid_loader):

        y = labels.cuda(non_blocking=True)
        data = [t.cuda(non_blocking=True) for t in data]
        x, coords = data[:2]

        if len(data) > 2: # has mask
            x = add_mask(x, data.pop(), 0)

        outputs = torch.zeros((5, h*w*t, target_size, target_size, target_size), dtype=dtype)
        #targets = torch.zeros((h*w*t, 9, 9, 9), dtype=torch.uint8)

        sample_loss = AverageMeter() if scoring and criterion is not None else None

        for b, coord in enumerate(coords.split(batch_size)):
            x1 = multicrop.crop3d_gpu(x, coord, sample_size, sample_size, sample_size, 1, True)
            x2 = multicrop.crop3d_gpu(x, coord, sub_sample_size, sub_sample_size, sub_sample_size, 3, True)

            if scoring:
                target = multicrop.crop3d_gpu(y, coord, target_size, target_size, target_size, 1, True)

            # compute output
            logit = model((x1, x2)) # nx5x9x9x9, target nx9x9x9
            output = F.softmax(logit, dim=1)

            # copy output
            start = b*batch_size
            end   = start + output.shape[0]
            outputs[:, start:end] = output.permute(1, 0, 2, 3, 4).cpu()

            #targets[start:end] = target.type(dtype).cpu()

            # measure accuracy and record loss
            if scoring and criterion is not None:
                loss = criterion(logit, target)
                sample_loss.update(loss.item(), target.size(0))

        outputs = outputs.view(5, h, w, t, 12, 12, 12).permute(0, 1, 4, 2, 5, 3, 6)
        outputs = outputs.reshape(5, h*12, w*12, t*12)
        outputs = outputs[:, :H, :W, :T].numpy()

        #targets = targets.view(h, w, t, 9, 9, 9).permute(0, 3, 1, 4, 2, 5).reshape(h*9, w*9, t*9)
        #targets = targets[:H, :W, :T].numpy()

        msg = 'Subject {}/{}, '.format(i+1, len(valid_loader))
        name = str(i)
        if names:
            name = names[i]
            msg += '{:>20}, '.format(name)

        if out_dir:
            np.save(os.path.join(out_dir, name + '_preds'), outputs)

        if scoring:
            labels  = labels.numpy()
            outputs = outputs.argmax(0)
            scores = dice(outputs, labels)

            #if criterion is not None:
            #    scores += sample_loss.avg,

            vals.update(np.array(scores))

            msg += ', '.join(['{}: {:.4f}'.format(k, v) for k, v in zip(keys, scores)])

        if verbose:
            logging.info(msg)

    if scoring:
        msg = 'Average scores: '
        msg += ', '.join(['{}: {:.4f}'.format(k, v) for k, v in zip(keys, vals.avg)])
        logging.info(msg)

    model.train()
    return vals.avg


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == '__main__':
    global args

    parser = argparse.ArgumentParser()
    #parser.add_argument('-cfg', '--cfg', default='deepmedic_ce_all', type=str)
    #parser.add_argument('-cfg', '--cfg', default='deepmedic_nr_ce_all', type=str)
    #parser.add_argument('-cfg', '--cfg', default='deepmedic_nr_ce', type=str)
    #parser.add_argument('-cfg', '--cfg', default='deepmedic_ce', type=str)
    #parser.add_argument('-cfg', '--cfg', default='deepmedic_ce_all', type=str)

    #parser.add_argument('-cfg', '--cfg', default='deepmedic_ce_50_50_all', type=str)
    parser.add_argument('-cfg', '--cfg', default='deepmedic_ce_50_50_c25_all', type=str)
    parser.add_argument('-gpu', '--gpu', default='0', type=str)

    args = parser.parse_args()
    args = Parser(args.cfg, log='test').add_args(args)

    #args.valid_list = 'valid_0.txt'
    #args.saving = False

    args.data_dir = '/usr/data/pkao/brats2018/testing'
    args.valid_list = 'test.txt'
    args.saving = True

    args.ckpt = 'model_last.tar'
    #args.ckpt = 'model_iter_227.tar'

    if args.saving:
        folder = os.path.splitext(args.valid_list)[0]
        out_dir = os.path.join('output', args.name, folder)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        args.out_dir = out_dir
    else:
        args.out_dir = ''

    args.scoring = not args.saving

    main()
