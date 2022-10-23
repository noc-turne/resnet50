import os
import argparse
import yaml
from addict import Dict
from PIL import Image
import torch
import torch.nn as nn

import models
from utils.dataloader import build_augmentation
from utils.misc import accuracy, check_keys

parser = argparse.ArgumentParser(description='ImageNet demo Example')
parser.add_argument('img', type=str, help='path to config file')
parser.add_argument('config', default='configs/resnet50.yaml',
                    type=str, help='path to config file')
parser.add_argument('ckpt', type=str, help='path to ckpt')
parser.add_argument('--topk', type=int, default=5, help='topk predict')


def main():
    args = parser.parse_args()
    args.config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    cfgs = Dict(args.config)

    model = models.__dict__[cfgs.net.arch](**cfgs.net.kwargs)
    model.cuda()

    print("creating model '{}'".format(cfgs.net.arch))

    assert os.path.isfile(args.ckpt), 'Not found ckpt model: {}'.format(args.ckpt)
    checkpoint = torch.load(args.ckpt)
    check_keys(model=model, checkpoint=checkpoint)
    model.load_state_dict(checkpoint['state_dict'])

    img = Image.open(args.img).convert('RGB')
    test_aug = build_augmentation(cfgs.dataset.test)
    input = test_aug(img)
    c, h, w = input.shape
    input = input.reshape(-1, c, h ,w)
    # test mode
    model.eval()
    with torch.no_grad():
        input = input.cuda()
        # compute output
        output = model(input)
        _, pred = output.topk(args.topk, 1, True, True)
        pred = pred.t().cpu().numpy()
        print("img '{}' pred: \n{}".format(args.img, pred))

if __name__ == '__main__':
    main()
