import os
import re

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

rank = int(os.environ['SLURM_PROCID'])
world_size = int(os.environ['SLURM_NTASKS'])
local_rank = int(os.environ['SLURM_LOCALID'])
node_list = str(os.environ['SLURM_NODELIST'])

node_parts = re.findall('[0-9]+', node_list)
host_ip = '{}.{}.{}.{}'.format(node_parts[1], node_parts[2], node_parts[3], node_parts[4])
port = "12221"
init_method = 'tcp://{}:{}'.format(host_ip, port)
print(init_method)

torch.distributed.init_process_group("nccl", init_method=init_method,
                                     world_size=world_size, rank=rank)
# os.environ['MASTER_ADDR'] = host_ip
# os.environ['MASTER_PORT'] = port
# torch.distributed.init_process_group("nccl", world_size=world_size, rank=rank)
torch.cuda.set_device(local_rank)

import pdb
pdb.set_trace()
model = nn.Conv2d(3, 8, (3, 3)).cuda()
model = DDP(model, device_ids=[local_rank])
model.train()
optimizer = torch.optim.SGD(model.parameters(), 0.1)

input = torch.rand(2, 3, 5, 5).float().cuda()
output = model(input)
loss = output.sum()

optimizer.zero_grad()
loss.backward()
optimizer.step()

torch.distributed.all_reduce(input)

torch.distributed.broadcast(input, src=0)
print("rank {} conv bias grad \n{}".format(rank, model.parameters().keys()))
