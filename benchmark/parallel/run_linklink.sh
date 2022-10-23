#!/bin/sh

p=$1
g=$(($2<8?$2:8))
#OMPI_MCA_mpi_warn_on_fork=0 srun --mpi=pmi2 -p $p -w SH-IDC1-10-5-38-253 --gres=gpu:$g -n $2 --ntasks-per-node=$g \
PYTHONNOUSERSITE=1 OMPI_MCA_mpi_warn_on_fork=0 srun --mpi=pmi2 -p $p --gres=gpu:$g -n $2 --ntasks-per-node=$g \
python -u benchmark_linklink.py --benchmark --max_iter=500 -a resnet50 -b 64 /mnt/lustre/share/images
