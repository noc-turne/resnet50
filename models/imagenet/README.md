# Classification
full doc：http://pape.parrots.sensetime.com/example_imagenet.html

## how to use

### 1. Prepare code and env
```bash
git clone git@gitlab.bj.sensetime.com:platform/ParrotsDL/parrots.example.git
cd parrots.example/models/imagenet/

# Pytorch env: pt0.4v1, pt1.0v1, pt1.1v1, pt1.2v1
# Parrots env: pat0.4.0rc1, pat0.5.0rc0
source pat0.5.0rc0
```
### 2. Training
If you use your own PAPE, modify `PYTHONPATH` in `train.sh` to your path.

```bash
# sh train.sh [ConfigFilePath] [JobName] [PartitionName] [GPUNum]
sh train.sh configs/resnet.yaml resnet50 Platform 8
```

### 3. Test
```bash
# sh test.sh [ConfigFilePath] [JobName] [PartitionName] [GPUNum]
sh test.sh configs/resnet.yaml resnet50 Platform 8
```
