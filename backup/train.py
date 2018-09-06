import argparse
import os
import shutil
import time
import logging
import random

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
cudnn.benchmark = True

import numpy as np

import models
from models import criterions
from data import datasets
from data.dataloader import DataLoader
from data.sampler import SSampler

from utils import Parser


parser = argparse.ArgumentParser()
parser.add_argument('-cfg', '--cfg', default='deepmedic3', type=str)

path = os.path.dirname(__file__)

## parse arguments
args = parser.parse_args()
args = Parser().add_args(args).add_cfg(args.cfg) #args.cfg, args)
ckpts = args.makedir()

args.gpu = str(args.gpu)

# setup logs
log_dir = os.path.join(path, 'logs')
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
log = os.path.join(log_dir, args.cfg + '.txt')
fmt = '%(asctime)s %(message)s'
logging.basicConfig(level=logging.INFO, format=fmt, filename=log)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
console.setFormatter(logging.Formatter(fmt))
logging.getLogger('').addHandler(console)

def main():
    # setup environments and seeds
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # setup networks
    Network = getattr(models, args.net)
    model = Network(**args.net_params)
    model = model.cuda()

    optimizer = getattr(torch.optim, args.opt)(
            model.parameters(), **args.opt_params)
    criterion = getattr(criterions, args.criterion)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_iter = checkpoint['iter']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optim_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))


    # Data loading code

    Dataset = getattr(datasets, args.dataset)

    # The loader will get 1000 patches from 50 subjects for each sub epoch
    # each subject sample 20 patches
    train_list = os.path.join(args.data_dir, 'file_list.txt')
    train_set = Dataset(train_list, root=args.data_dir,
            for_train=True, sample_size=args.patch_per_sample)

    num_iters  = args.num_iters or len(train_set) * args.num_epochs
    num_iters -= args.start_iter

    train_sampler = SSampler(len(train_set), num_iters)

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        collate_fn=train_set.collate, sampler=train_sampler,
        num_workers=args.workers, pin_memory=True, prefetch=False)

    #repeat = False
    #valid_dir = args.data_dir
    #valid_list = os.path.join(args.data_dir, 'valid_list.txt')
    #valid_set = Dataset(valid_list, root=args.data_dir, for_train=False)
    #valid_loader = DataLoader(
    #    valid_set,
    #    batch_size=1, shuffle=False,
    #    collate_fn=valid_set.collate,
    #    num_workers=2, pin_memory=True, prefetch=repeat)


    logging.info('-------------- New training session ----------------')

    start = time.time()

    enum_batches = len(train_set)/args.batch_size

    args.schedule  = {int(k*enum_batches): v for k, v in args.schedule.items()}
    args.save_freq = int(enum_batches * args.save_freq)


    ## this is 0? in their configuration file
    #stop_class_balancing = int(args.stop_class_balancing * enum_batches)
    #weight = None

    losses = AverageMeter()
    torch.set_grad_enabled(True)

    for i, (data, label) in enumerate(train_loader, args.start_iter):

        adjust_learning_rate(optimizer, i)

        # look at dataset class
        #weight = data.pop()
        #if i < stop_class_balancing:
        #    alpha = float(i)/stop_class_balancing
        #    weight = alpha + (1.0 - alpha)*weight # alpha*y2 + (1.0 - alpha)*y1
        #    weight = weight.cuda(non_blocking=True)
        #else:
        #    weight = None
        #data = [d.cuda(non_blocking=True) for d in data]

        for x1, x2, target in zip(*[d.split(args.batch_size) for d in data]):

            x1, x2, target = [t.cuda(non_blocking=True) for t in (x1, x2, target)]

            # compute output
            output = model((x1, x2)) # nx5x9x9x9, target nx9x9x9
            loss = criterion(output, target)
            #loss = criterion(output, target, weight)

            # measure accuracy and record loss
            losses.update(loss.item(), target.size(0))

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (i+1) % args.save_freq == 0:
            file_name = os.path.join(ckpts, 'model_iter_{}.tar'.format(i+1))
            torch.save({
                'iter': i+1,
                'state_dict': model.state_dict(),
                'optim_dict': optimizer.state_dict(),
                },
                file_name)

        msg = 'Iter {0:}, Epoch {1:.4f}, Loss {2:.4f}'.format(
                i+1, (i+1)/enum_batches, losses.avg)
        logging.info(msg)

        losses.reset()

    file_name = os.path.join(ckpts, 'model_last.tar')
    torch.save({
        'iter': i+1,
        'state_dict': model.state_dict(),
        'optim_dict': optimizer.state_dict(),
        },
        file_name)

    msg = 'total time: {} minutes'.format((time.time() - start)/60)
    logging.info(msg)



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


def adjust_learning_rate(optimizer, epoch):
    # reduce learning rate by a factor of 10
    if epoch+1 in args.schedule:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.1


if __name__ == '__main__':
    main()
