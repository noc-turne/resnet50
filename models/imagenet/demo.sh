partition=$1
img=$2
cfg=$3
ckpt=$4


OMPI_MCA_mpi_warn_on_fork=0 GLOG_vmodule=MemcachedClient=-1 \
srun --mpi=pmi2 -p $partition \
    --gres=gpu:1 -n1 --ntasks-per-node=1 \
    python -u demo.py $img $cfg $ckpt
