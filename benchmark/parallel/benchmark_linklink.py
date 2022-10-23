import argparse
import os
import random
import shutil
import time
import warnings
import numpy as np

import torch
import torch.nn as nn
import torch.backends as backends
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.models as models

import linklink as link
from utils.linklink_utils import dist_init, reduce_gradients, DistModule
from utils.memcached_dataset import McDataset

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='Benchmark PAPE parallel Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=20, type=int,
                    metavar='N', help='print frequency')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')

# parallel args
parser.add_argument('--parallel', default='overlap',
                    choices=['naive', 'overlap', 'non_overlap'],
                    help='parallel type')
parser.add_argument('--bucket_size', default=1., type=float,
                    help='parallel bucket_size(MB)')
parser.add_argument('--syncbn', action='store_true', help='use syncbn')

# benchmark args
parser.add_argument('--benchmark', action='store_true',
                    help='benchmark mode, will use dummy input data')
parser.add_argument('--max_iter', default=2000, type=int,
                    help='max benchmark iteration')


def main():
    args = parser.parse_args()
    args.rank, args.world_size = dist_init()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        backends.cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True)
    else:
        print("=> creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch]()

    model.cuda()

    if args.syncbn:
        model = torch.nn.SyncBatchNrom.convert_sync_batchnorm(model)

    model = DistModule(model)

    if args.rank == 0:
        print(model)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    if args.rank == 0:
        print(criterion)

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    best_acc1 = 0.0
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    backends.cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    train_meta_file = os.path.join(args.data, 'meta/train.txt')
    valdir = os.path.join(args.data, 'val')
    val_meta_file = os.path.join(args.data, 'meta/val.txt')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = McDataset(
        traindir, train_meta_file,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    with open(val_meta_file) as f:
        val_items = [line.strip() for line in f.readlines()]
    val_full_size = len(val_items)
    val_remain_size = val_full_size % (args.batch_size * args.world_size)
    val_size = val_full_size - val_remain_size

    val_dataset = McDataset(
        valdir, val_items[0:val_size],
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))

    val_sampler = None
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=(val_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=val_sampler)

    val_remain_dataset = McDataset(
        valdir, val_items[val_size:],
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))
    val_remain_loader = torch.utils.data.DataLoader(
        val_remain_dataset, batch_size=args.batch_size, shuffle=None,
        num_workers=args.workers, pin_memory=True)

    if args.evaluate:
        validate(val_loader, model, criterion, args)
        return

    for epoch in range(args.start_epoch, args.epochs):
        # train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args)

        if args.benchmark:
            return
        # evaluate on validation set
        acc1 = validate(val_loader, val_remain_loader, val_full_size, model, criterion, args)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        if args.rank == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer': optimizer.state_dict(),
            }, is_best)

    link.finalize()


def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f', 50)
    data_time = AverageMeter('Data', ':6.3f', 50)
    losses = AverageMeter('Loss', ':.4f', 10)
    top1 = AverageMeter('Acc@1', ':6.2f', 10)
    top5 = AverageMeter('Acc@5', ':6.2f', 10)

    if args.benchmark:
        dummy_input = torch.rand([args.batch_size, 3, 224, 224]).cuda()
        dummy_target = torch.randint(1000, (args.batch_size,), dtype=torch.long).cuda()
        max_iter = args.max_iter
        progress = ProgressMeter(max_iter, batch_time, prefix="Benchmark: ")
    else:
        train_iter = iter(train_loader)
        max_iter = len(train_loader)
        progress = ProgressMeter(max_iter, batch_time, data_time, losses, top1,
                                 top5, prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i in range(max_iter):
        if args.benchmark:
            input, target = dummy_input, dummy_target
        else:
            input, target = train_iter.next()

        input = input.cuda()
        target = target.cuda()

        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item())
        top1.update(acc1[0].item())
        top5.update(acc5[0].item())

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward(torch.tensor(1.0/args.world_size).cuda())
        reduce_gradients(model)    # only need in DistributedNaiveModel
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if args.rank == 0 and (i % args.print_freq == 0 or i == max_iter - 1):
            progress.print(i)


def validate(val_loader, val_remain_loader, val_full_size, model, criterion, args):
    losses = AverageMeter('Loss', ':.4f')
    top_sum = torch.Tensor([0, 0]).long()

    # switch to evaluate mode
    model.eval()

    # start_time = time.time()
    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            input = input.cuda()
            target = target.cuda()

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5, acc1_cnt, acc5_cnt = accuracy(output, target, topk=(1, 5), need_raw=True)
            losses.update(loss.item())
            top_sum[0] += acc1_cnt[0].item()
            top_sum[1] += acc5_cnt[0].item()

            # measure elapsed time
            # batch_time = time.time() - end
            end = time.time()

            if args.rank == 0 and i % args.print_freq == 0:
                batch_size = target.size(0)
                print('Test [{}/{}] {} Acc@1 {:.3f} ({}/{}) Acc@5 {:.3f} ({}/{})'.format(
                          i, len(val_loader), losses,
                          acc1[0].item(), int(acc1_cnt[0].item()), batch_size,
                          acc5[0].item(), int(acc5_cnt[0].item()), batch_size))

        dist.all_reduce(top_sum, op=dist.ReduceOp.SUM)

        for i, (input, target) in enumerate(val_remain_loader):
            input = input.cuda()
            target = target.cuda()

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5, acc1_cnt, acc5_cnt = accuracy(output, target, topk=(1, 5), need_raw=True)
            losses.update(loss.item())
            top_sum[0] += acc1_cnt[0].item()
            top_sum[1] += acc5_cnt[0].item()
            if args.rank == 0:
                batch_size = target.size(0)
                print('Remain Test [{}/{}] {} Acc@1 {:.3f} ({}/{}) Acc@5 {:.3f} ({}/{})'.format(
                          i, len(val_remain_loader), losses,
                          acc1[0].item(), int(acc1_cnt[0].item()), batch_size,
                          acc5[0].item(), int(acc5_cnt[0].item()), batch_size))

        top_acc = top_sum.float() / val_full_size * 100
        if args.rank == 0:
            print(' * All Loss {} Acc@1 {:.3f} ({}/{}) Acc@5 {:.3f} ({}/{})'.format(losses,
                  top_acc[0], top_sum[0], val_full_size,
                  top_acc[1], top_sum[1], val_full_size))

    return top_acc[0]


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value
       When length <=0 , save all history data """

    def __init__(self, name, fmt=':f', length=0):
        self.name = name
        self.fmt = fmt
        self.length = length
        self.reset()

    def reset(self):
        if self.length > 1:
            self.history = []
        elif self.length <= 0:
            self.count = 0
            self.sum = 0
        self.avg = 0
        self.val = 0

    def update(self, val):
        self.val = val
        if self.length > 1:
            self.history.append(val)
            if len(self.history) > self.length:
                del self.history[0]
            self.avg = np.mean(self.history)
        elif self.length <= 0:
            self.sum += val
            self.count += 1
            self.avg = self.sum / self.count

    def __str__(self):
        if self.length == 1:
            fmtstr = '{name} {val' + self.fmt + '}'
        else:
            fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, *meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def print(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,), need_raw=False):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        res_raw = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res_raw.append(correct_k)
            res.append(correct_k.mul(100.0 / batch_size))
        if need_raw:
            res.extend(res_raw)
        return res


if __name__ == '__main__':
    main()
