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
            for_train=False, crop=False,
            transforms=args.test_transforms)
    valid_loader = DataLoader(
        valid_set, batch_size=1, shuffle=False,
        collate_fn=valid_set.collate,
        num_workers=4, pin_memory=True)

    start = time.time()
    with torch.no_grad():
        scores = validate(valid_loader, model, args.batch_size,
                args.out_dir, valid_set.names)

    msg = 'total time {:.4f} minutes'.format((time.time() - start)/60)
    logging.info(msg)

def test(valid_loader, model, batch_size,
        out_dir='', names=None, verbose=True):
    # need a seperate test function if we want ensemble many models
    # saving probs instead of preds

    h, w, t = 27, 27, 18
    H, W, T = 240, 240, 155
    dtype = torch.uint8

    model.eval()

    for i, (data, labels) in enumerate(valid_loader):

        x, coords = [t.cuda(non_blocking=True) for t in data]

        outputs = torch.zeros((5, h*w*t, 9, 9, 9), dtype=dtype)

        sample_loss = AverageMeter() if scoring and criterion is not None else None

        for b, coord in enumerate(coords.split(batch_size)):
            x1 = multicrop.crop3d_gpu(x, coord, 25, 25, 25, 1, True)
            x2 = multicrop.crop3d_gpu(x, coord, 19, 19, 19, 3, True)

            # compute output
            logit = model((x1, x2)) # nx5x9x9x9, target nx9x9x9
            logit = F.softmax(logit, dim=1)

            # copy output
            start = b*batch_size
            end   = start + output.shape[0]
            outputs[:, start:end] = output.permute(1, 0, 2, 3, 4).type(dtype).cpu()

        outputs = outputs.view(5, h, w, t, 9, 9, 9).permute(0, 1, 4, 2, 5, 3, 6)
        outputs = outputs.reshape(5, h*9, w*9, t*9)
        outputs = outputs[:, :H, :W, :T].numpy()

        msg = 'Subject {}/{}, '.format(i+1, len(valid_loader))
        name = str(i)
        if names:
            name = names[i]
            msg += '{:>20}, '.format(name)

        if out_dir:
            np.save(os.path.join(out_dir, name + '_preds'), outputs)


        if verbose:
            logging.info(msg)

    model.train()

def validate(valid_loader, model, batch_size,
        out_dir='', names=None, scoring=True, verbose=True):

    h, w, t = 27, 27, 18
    H, W, T = 240, 240, 155
    dtype = torch.uint8

    model.eval()
    criterion = F.cross_entropy

    vals = AverageMeter()
    for i, (data, labels) in enumerate(valid_loader):

        y = labels.cuda(non_blocking=True)
        x, coords = [t.cuda(non_blocking=True) for t in data]

        outputs = torch.zeros((h*w*t, 9, 9, 9), dtype=dtype)
        #targets = torch.zeros((h*w*t, 9, 9, 9), dtype=dtype)

        sample_loss = AverageMeter() if scoring and criterion is not None else None

        for b, coord in enumerate(coords.split(batch_size)):
            x1 = multicrop.crop3d_gpu(x, coord, 25, 25, 25, 1, True)
            x2 = multicrop.crop3d_gpu(x, coord, 19, 19, 19, 3, True)
            target = multicrop.crop3d_gpu(y, coord, 9, 9, 9, 1, True)

            # compute output
            logit = model((x1, x2)) # nx5x9x9x9, target nx9x9x9
            _, output = logit.max(dim=1) # nx9x9x9

            # copy output
            start = b*batch_size
            end   = start + output.shape[0]
            outputs[start:end] = output.type(dtype).cpu()

            #targets[start:end] = target.type(dtype).cpu()

            # measure accuracy and record loss
            if scoring and criterion is not None:
                loss = criterion(logit, target)
                sample_loss.update(loss.item(), target.size(0))

        outputs = outputs.view(h, w, t, 9, 9, 9).permute(0, 3, 1, 4, 2, 5)
        outputs = outputs.reshape(h*9, w*9, t*9)
        outputs = outputs[:H, :W, :T].numpy()

        msg = 'Subject {}/{}, '.format(i+1, len(valid_loader))
        name = str(i)
        if names:
            name = names[i]
            msg += '{:>20}, '.format(name)

        if out_dir:
            np.save(os.path.join(out_dir, name + '_preds'), outputs)

        if scoring:
            labels  = labels.numpy()
            scores = dice(outputs, labels)

            #if criterion is not None:
            #    scores += sample_loss.avg,

            vals.update(np.array(scores))

            msg += ', '.join(['{}: {:.4f}'.format(k, v) for k, v in zip(keys, scores)])

        if verbose:
            logging.info(msg)

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
    parser.add_argument('-cfg', '--cfg', default='deepmedic_ce_all', type=str)
    parser.add_argument('-gpu', '--gpu', default='3', type=str)
    args = parser.parse_args()

    args.cfg = 'deepmedic_nr'
    args.gpu = str(args.gpu)
    args.gpu = '3'

    args = Parser(args.cfg, log='test').add_args(args)

    args.valid_list = 'train_0.txt'

    args.data_dir = '/home/thuyen/Data/brats17/Brats17ValidationData'
    args.valid_list = 'test.txt'

    args.ckpt = 'model_last.tar'
    #args.ckpt = 'model_iter_227.tar'

    folder = os.path.splitext(args.valid_list)[0]
    out_dir = os.path.join('output', args.name, folder)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    args.out_dir = out_dir

    main()
