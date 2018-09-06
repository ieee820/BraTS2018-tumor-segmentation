import argparse
import os
import shutil
import time
import logging
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
cudnn.benchmark = True

import numpy as np

import models
from models import criterions
from data import datasets
from data.dataloader import DataLoader

from utils import Parser

path = os.path.dirname(__file__)


cfg_name = 'deepmedic'

mode = 'train'

args = Parser().add_cfg(cfg_name)
ckpts = args.getdir()

args.gpu = str(args.gpu)
args.gpu = '0'

out_dir = os.path.join('output', cfg_name, mode)
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

# setup logs
log = os.path.join(out_dir, 'log.txt')
fmt = '%(asctime)s %(message)s'
logging.basicConfig(level=logging.INFO, format=fmt, filename=log)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
console.setFormatter(logging.Formatter(fmt))
logging.getLogger('').addHandler(console)

eps = 1e-5
def f1_score(o, t):
    num = 2*(o*t).sum() + eps
    den = o.sum() + t.sum() + eps
    return num/den

def dice(output, target):
    ret = []
    ret += f1_score(output > 0, target > 0),
    for c in range(1, 5):
        ret += f1_score(output == c, target == c),
    return ret

keys = 'dice0', 'dice1', 'dice2', 'dice3', 'dice4', 'loss'
def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    # setup networks
    Network = getattr(models, args.net)
    model = Network(**args.net_params)
    model = model.cuda()
    #criterion = getattr(criterions, args.criterion)
    criterion = F.cross_entropy

    model_file = os.path.join(ckpts, 'model_last.tar')
    checkpoint = torch.load(model_file)
    model.load_state_dict(checkpoint['state_dict'])

    valid_list = os.path.join(args.data_dir, 'file_list.txt')
    Dataset = getattr(datasets, args.dataset)

    repeat = False
    valid_dir = args.data_dir
    valid_set = Dataset(valid_list, root=valid_dir, for_train=False)
    valid_loader = DataLoader(
        valid_set,
        batch_size=1, shuffle=False,
        collate_fn=valid_set.collate,
        num_workers=2, pin_memory=True, prefetch=repeat)

    start = time.time()
    model.eval()
    with torch.no_grad():
        scores = validate(valid_loader, model, criterion, out_dir, valid_set.names)

    msg = 'total time {:.4f} minutes'.format((time.time() - start)/60)
    logging.info(msg)


def validate(valid_loader, model, criterion=None,
        out_dir='', names=None, scoring=True):
    h, w, t = 27, 27, 18
    H, W, T = 240, 240, 155
    dtype = torch.uint8

    vals = AverageMeter()
    for i, (data, labels) in enumerate(valid_loader):

        outputs = torch.zeros((h*w*t, 9, 9, 9), dtype=dtype)

        sample_loss = AverageMeter() if scoring and criterion is not None else None

        for b, (x1, x2, target) in enumerate(zip(*[d.split(args.batch_size) for d in data])):

            x1, x2, target = [t.cuda(non_blocking=True) for t in (x1, x2, target)]

            # compute output
            logit = model((x1, x2)) # nx5x9x9x9, target nx9x9x9
            _, output = logit.max(dim=1) # nx9x9x9

            # copy output
            start = b*args.batch_size
            end   = start + output.shape[0]
            outputs[start:end] = output.type(dtype).cpu()

            # measure accuracy and record loss
            if scoring and criterion is not None:
                loss = criterion(logit, target)
                sample_loss.update(loss.item(), target.size(0))

        outputs = outputs.view(h, w, t, 9, 9, 9).permute(0, 3, 1, 4, 2, 5)
        outputs = outputs.reshape(h*9, w*9, t*9)
        outputs = outputs[:H, :W, :T].numpy()

        msg = ''
        if out_dir:
            name = names[i] if names else str(i)
            np.save(os.path.join(out_dir, name + '_preds'), outputs)
            msg += '{}, '.format(name)

        if scoring:
            labels  = labels.numpy()
            scores = dice(outputs, labels)

            if criterion is not None:
                scores += sample_loss.avg,

            vals.update(np.array(scores))

            msg += ', '.join(['{}: {:.4f}'.format(k, v) for k, v in zip(keys, scores)])

        logging.info(msg)
        exit(0)

    msg = 'Average scores: '
    msg += ', '.join(['{}: {:.4f}'.format(k, v) for k, v in zip(keys, vals.avg)])
    logging.info(msg)

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
    main()
