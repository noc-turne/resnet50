mkdir -p log
now=$(date +"%Y%m%d_%H%M%S")

cfg=$1
jobname=$2
partition=$3
gpus=$4

g=$(($gpus<8?$gpus:8))

OMPI_MCA_mpi_warn_on_fork=0 GLOG_vmodule=MemcachedClient=-1 \
srun --mpi=pmi2 -p $partition --job-name=$jobname \
    --gres=gpu:$g -n$gpus --ntasks-per-node=$g \
    python -u main.py --config $cfg \
    2>&1 | tee log/train_$jobname.log-$now
