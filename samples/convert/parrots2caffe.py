import torch
from parrotsconvert.caffe import BackendSet, CaffeNet

import torchvision.models as models

net = models.__dict__['resnet18']()
net.eval()

caffe_net = CaffeNet(net, torch.randn(1, 3, 224, 224), BackendSet.SENSENET)
caffe_net.merge_bn()
caffe_net.dump_model('./resnet')
