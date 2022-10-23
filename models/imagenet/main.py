import os
import shutil
import argparse
import random
import time
import yaml
import json
import socket
import logging
from addict import Dict
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.distributed as dist
import torch.optim
from torch.backends import cudnn
from torch.nn.parallel import DistributedDataParallel as DDP

import models
from utils.dataloader import build_dataloader
from utils.misc import accuracy, check_keys, AverageMeter, ProgressMeter
from utils.loss import LabelSmoothLoss

parser = argparse.ArgumentParser(description='ImageNet Training Example')
parser.add_argument('--config',
                    default='configs/resnet50.yaml',
                    type=str,
                    help='path to config file')
parser.add_argument('--test',
                    dest='test',
                    action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--dummy_test',
                    dest='dummy_test',
                    action='store_true',
                    help='dummy data for speed evaluation')
parser.add_argument('--taskid', default='None', type=str, help='pavi taskid')
parser.add_argument('--port',
                    default=12345,
                    type=int,
                    metavar='P',
                    help='master port')
parser.add_argument('--test_freq',
                    default=None,
                    type=str,
                    help='test frequency')
parser.add_argument('--save_path',
                    default=None,
                    type=str,
                    help='checkpoint path')
parser.add_argument('--resume',
                    default=None,
                    type=str,
                    help='resume checkpoint')
parser.add_argument('--reader',
                    type=str,
                    default='MemcachedReader',
                    choices=['MemcachedReader', 'CephReader', 'DirectReader'],
                    help='io backend')
parser.add_argument('--pavi',
                    default=None,
                    type=str,
                    help='pavi use and pavi project')

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger()
logger_all = logging.getLogger('all')


def main():
    args = parser.parse_args()
    args.config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    cfgs = Dict(args.config)
    if 'SLURM_PROCID' in os.environ.keys():
        args.rank = int(os.environ['SLURM_PROCID'])
        args.world_size = int(os.environ['SLURM_NTASKS'])
        args.local_rank = int(os.environ['SLURM_LOCALID'])
        os.environ['MASTER_ADDR'] = socket.gethostbyname(
            socket.getfqdn(socket.gethostname()))
        os.environ['MASTER_PORT'] = str(args.port)
    else:
        args.rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
        args.world_size = int(os.environ['OMPI_COMM_WORLD_SIZE'])
        args.local_rank = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])

    os.environ['WORLD_SIZE'] = str(args.world_size)
    os.environ['RANK'] = str(args.rank)

    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    torch.cuda.set_device(args.local_rank)

    if args.rank == 0:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.ERROR)
    logger_all.setLevel(logging.INFO)

    logger_all.info("rank {} of {} jobs, in {}".format(args.rank,
                                                       args.world_size,
                                                       socket.gethostname()))

    dist.barrier()

    logger.info("config\n{}".format(
        json.dumps(cfgs, indent=2, ensure_ascii=False)))

    if cfgs.get('seed', None):
        random.seed(cfgs.seed)
        torch.manual_seed(cfgs.seed)
        torch.cuda.manual_seed(cfgs.seed)
        cudnn.deterministic = True

    model = models.__dict__[cfgs.net.arch](**cfgs.net.kwargs)
    model.cuda()

    logger.info("creating model '{}'".format(cfgs.net.arch))

    model = DDP(model, device_ids=[args.local_rank])
    logger.info("model\n{}".format(model))

    if cfgs.get('label_smooth', None):
        criterion = LabelSmoothLoss(cfgs.trainer.label_smooth,
                                    cfgs.net.kwargs.num_classes).cuda()
    else:
        criterion = nn.CrossEntropyLoss().cuda()
    logger.info("loss\n{}".format(criterion))

    optimizer = torch.optim.SGD(model.parameters(),
                                **cfgs.trainer.optimizer.kwargs)
    logger.info("optimizer\n{}".format(optimizer))

    cudnn.benchmark = True

    args.start_epoch = -cfgs.trainer.lr_scheduler.get('warmup_epochs', 0)
    args.max_epoch = (int(os.getenv("maxstep")) if os.getenv("maxstep")
                      is not None else cfgs.trainer.max_epoch)
    args.test_freq = (args.test_freq
                      if args.test_freq else cfgs.trainer.test_freq)
    args.log_freq = cfgs.trainer.log_freq

    best_acc1 = 0.0
    resume_model = args.resume if args.resume else cfgs.saver.resume_model
    if resume_model:
        assert os.path.isfile(
            resume_model), 'Not found resume model: {}'.format(resume_model)
        checkpoint = torch.load(resume_model)
        check_keys(model=model, checkpoint=checkpoint)
        model.load_state_dict(checkpoint['state_dict'])
        args.start_epoch = checkpoint['epoch']
        best_acc1 = checkpoint['best_acc1']
        optimizer.load_state_dict(checkpoint['optimizer'])
        args.taskid = checkpoint['taskid']
        logger.info("resume training from '{}' at epoch {}".format(
            cfgs.saver.resume_model, checkpoint['epoch']))
    elif cfgs.saver.pretrain_model:
        assert os.path.isfile(
            cfgs.saver.pretrain_model), 'Not found pretrain model: {}'.format(
                cfgs.saver.pretrain_model)
        checkpoint = torch.load(cfgs.saver.pretrain_model)
        check_keys(model=model, checkpoint=checkpoint)
        model.load_state_dict(checkpoint['state_dict'])
        logger.info("pretrain training from '{}'".format(
            cfgs.saver.pretrain_model))

    save_dir = args.save_path if args.save_path else cfgs.saver.save_dir

    logger.info("sava_path'{}'".format(save_dir))

    if args.rank == 0:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            logger.info("create checkpoint folder {}".format(save_dir))

    # Data loading code
    train_loader, train_sampler, test_loader, _ = build_dataloader(
        cfgs.dataset, args.world_size, args.reader)

    # test mode
    if args.test:
        test(test_loader, model, criterion, args)
        return

    # choose scheduler
    lr_scheduler = torch.optim.lr_scheduler.__dict__[
        cfgs.trainer.lr_scheduler.type](optimizer if isinstance(
            optimizer, torch.optim.Optimizer) else optimizer.optimizer,
                                        **cfgs.trainer.lr_scheduler.kwargs,
                                        last_epoch=args.start_epoch - 1)

    monitor_writer = None

    if args.rank == 0 and (cfgs.get('monitor', None) or args.pavi):
        if args.pavi:
            monitor_kwargs = {'task': cfgs.net.arch, 'project': args.pavi}
            from pavi import SummaryWriter
            monitor_writer = SummaryWriter(session_text=yaml.dump(args.config),
                                       **monitor_kwargs)
            args.taskid = monitor_writer.taskid
        else:
            monitor_kwargs = cfgs.monitor.kwargs
            if hasattr(args, 'taskid'):
                monitor_kwargs['taskid'] = args.taskid
            elif hasattr(cfgs.monitor, '_taskid'):
                monitor_kwargs['taskid'] = cfgs.monitor._taskid


    [hook.before_run(
        len_loader=len(train_loader),
        model=model
        )
     for hook in getattr(torch, '_algolib_hooks', [])]

    # training
    for epoch in range(args.start_epoch, args.max_epoch):
        [
            hook.before_epoch(current_epoch=epoch)
            for hook in getattr(torch, '_algolib_hooks', [])
        ]
        train_sampler.set_epoch(epoch)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args,
              monitor_writer)

        if (epoch + 1) % args.test_freq == 0 or epoch + 1 == args.max_epoch:
            # evaluate on validation set

            loss, acc1, acc5 = test(test_loader, model, criterion, args)

            if args.rank == 0:
                if monitor_writer:
                    monitor_writer.add_scalar('Accuracy_Test_top1', acc1,
                                              len(train_loader) * epoch)
                    monitor_writer.add_scalar('Accuracy_Test_top5', acc5,
                                              len(train_loader) * epoch)
                    monitor_writer.add_scalar('Test_loss', loss,
                                              len(train_loader) * epoch)

            if args.rank == 0:
                checkpoint = {
                    'epoch': epoch + 1,
                    'arch': cfgs.net.arch,
                    'state_dict': model.state_dict(),
                    'best_acc1': best_acc1,
                    'optimizer': optimizer.state_dict(),
                    'taskid': args.taskid
                }

                ckpt_path = os.path.join(
                    save_dir,
                    cfgs.net.arch + '_ckpt_epoch_{}.pth'.format(epoch))
                best_ckpt_path = os.path.join(save_dir,
                                              cfgs.net.arch + '_best.pth')
                torch.save(checkpoint, ckpt_path)
                if acc1 > best_acc1:
                    best_acc1 = acc1
                    shutil.copyfile(ckpt_path, best_ckpt_path)
        else:
            loss, acc1, acc5 = (None, ) * 3

        [hook.after_epoch(current_epoch=epoch,
                          pavi_args=dict(
                              iteration=len(train_loader) * epoch,
                              enable=((epoch + 1) % args.test_freq == 0 or
                                      epoch + 1 == args.max_epoch),
                              values=dict(Accuracy_Test_top1=acc1,
                                          Accuracy_Test_top5=acc5,
                                          Test_loss=loss)))
         for hook in getattr(torch, '_algolib_hooks', [])]

        lr_scheduler.step()
    [hook.after_run() for hook in getattr(torch, '_algolib_hooks', [])]


def train(train_loader, model, criterion, optimizer, epoch, args,
          monitor_writer):
    batch_time = AverageMeter('Time', ':.3f', 200)
    data_time = AverageMeter('Data', ':.3f', 200)

    losses = AverageMeter('Loss', ':.4f', 50)
    top1 = AverageMeter('Acc@1', ':.2f', 50)
    top5 = AverageMeter('Acc@5', ':.2f', 50)

    memory = AverageMeter('Memory(MB)', ':.0f')
    progress = ProgressMeter(len(train_loader),
                             batch_time,
                             data_time,
                             losses,
                             top1,
                             top5,
                             memory,
                             prefix="Epoch: [{}/{}]".format(
                                 epoch + 1, args.max_epoch))

    # switch to train mode
    model.train()
    end = time.time()
    loader_length = len(train_loader)
    if args.dummy_test:
        input_, target_ = next(iter(train_loader))
        train_loader = [(i, i) for i in range(len(train_loader))].__iter__()
    for i, (input, target) in enumerate(train_loader):
        data_list = [input, target]
        [hook.before_iter(
            data=data_list,
        ) for hook in getattr(torch, '_algolib_hooks', [])]
        input, target = data_list
        # measure data loading time
        data_time.update(time.time() - end)

        if args.dummy_test:
            input = input_.detach()
            input.requires_grad = True
            target = target_
        input = input.cuda()
        target = target.cuda()

        # compute output
        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        stats_all = torch.tensor([loss.item(), acc1[0].item(),
                                  acc5[0].item()]).float().cuda()
        dist.all_reduce(stats_all)
        stats_all /= args.world_size

        losses.update(stats_all[0].item())
        top1.update(stats_all[1].item())
        top5.update(stats_all[2].item())
        memory.update(torch.cuda.max_memory_allocated() / 1024 / 1024)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.log_freq == 0:
            progress.display(i)
            cur_iter = epoch * loader_length + i
            if args.rank == 0 and monitor_writer:
                monitor_writer.add_scalar('Train_Loss', losses.avg, cur_iter)
                monitor_writer.add_scalar('Accuracy_train_top1', top1.avg,
                                          cur_iter)
                monitor_writer.add_scalar('Accuracy_train_top5', top5.avg,
                                          cur_iter)
        [hook.after_iter(iteration=i,
                         pavi_args=dict(
                             iteration=cur_iter,
                             enable=(i % args.log_freq == 0),
                             values=dict(Accuracy_train_top1=top1.avg,
                                         Accuracy_train_top5=top5.avg,
                                         Train_Loss=losses.avg)))
         for hook in getattr(torch, '_algolib_hooks', [])]


def test(test_loader, model, criterion, args):
    logger = logging.getLogger()
    logger_all = logging.getLogger('all')
    if args.rank == 0:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.ERROR)
    logger_all.setLevel(logging.INFO)
    batch_time = AverageMeter('Time', ':.3f', 10)
    losses = AverageMeter('Loss', ':.4f', -1)
    top1 = AverageMeter('Acc@1', ':.2f', -1)
    top5 = AverageMeter('Acc@5', ':.2f', -1)
    stats_all = torch.Tensor([0, 0, 0]).long()
    progress = ProgressMeter(len(test_loader),
                             batch_time,
                             losses,
                             top1,
                             top5,
                             prefix="Test: ")

    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        end = time.time()
        if args.dummy_test:
            input_, target_ = next(iter(test_loader))
            test_loader = [(i, i) for i in range(len(test_loader))]
        for i, (input, target) in enumerate(test_loader):
            if args.dummy_test:
                input = input_
                target = target_
            input = input.cuda()
            target = target.cuda()

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5), raw=True)

            losses.update(loss.item())
            top1.update(acc1[0].item() * 100.0 / target.size(0))
            top5.update(acc5[0].item() * 100.0 / target.size(0))

            stats_all.add_(
                torch.tensor([acc1[0].item(), acc5[0].item(),
                              target.size(0)]).long())

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.log_freq == 0:
                progress.display(i)

        logger_all.info(
            ' Rank {} Loss {:.4f} Acc@1 {} Acc@5 {} total_size {}'.format(
                args.rank, losses.avg, stats_all[0].item(),
                stats_all[1].item(), stats_all[2].item()))

        loss = torch.tensor([losses.avg])
        dist.all_reduce(loss.cuda())
        loss_avg = loss.item() / args.world_size
        dist.all_reduce(stats_all.cuda())
        acc1 = stats_all[0].item() * 100.0 / stats_all[2].item()
        acc5 = stats_all[1].item() * 100.0 / stats_all[2].item()

        logger.info(
            ' * All Loss {:.4f} Acc@1 {:.3f} ({}/{}) Acc@5 {:.3f} ({}/{})'.
            format(loss_avg, acc1, stats_all[0].item(), stats_all[2].item(),
                   acc5, stats_all[1].item(), stats_all[2].item()))

        logger.info(
            '__benchmark_result__ * All Loss {:.4f} Acc@1 {:.3f} ({}/{}) Acc@5 {:.3f} ({}/{})'
            .format(loss_avg, acc1, stats_all[0].item(), stats_all[2].item(),
                    acc5, stats_all[1].item(), stats_all[2].item()))

    return loss_avg, acc1, acc5


if __name__ == '__main__':
    main()
