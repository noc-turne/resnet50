set -x
set -o pipefail
set -e

# 1. build file folder for save log,format: algolib_gen/frame
mkdir -p algolib_gen/example

# 2. set time
now=$(date +"%Y%m%d_%H%M%S")

# 3. set env
path=$PWD
if [[ "$path" =~ "submodules/example" ]]
then 
    pyroot=$path
    algolib_root=$path/../..
    init_path=$path/..
else
    pyroot=$path/submodules/example
    algolib_root=$path
    init_path=$path/submodules
fi
export FRAME_NAME=example # customize for each frame
export MODEL_NAME=$3
cfg=$pyroot/algolib/configs/${MODEL_NAME}.yaml
export PYTHONPATH=$algolib_root:$pyroot:$PYTHONPATH

# init_path
export PYTHONPATH=$init_path/common/sites/:$PYTHONPATH # necessary for init

# 4. build necessary parameter
partition=$1
g=$(($2<8?$2:8))
array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:3:$len}
SRUN_ARGS=${SRUN_ARGS:-""}

port=`expr $RANDOM % 10000 + 20000`

# 5. model choice
mkdir -p algolib_gen/example/${MODEL_NAME}/
export PARROTS_DEFAULT_LOGGER=FALSE
if [[ $3 =~ "sync" ]]; then
    PARROTS_EXEC_MODE=SYNC OMPI_MCA_mpi_warn_on_fork=0 GLOG_vmodule=MemcachedClient=-1 \
    srun --mpi=pmi2 -p $partition --job-name=example_${MODEL_MODEL_NAME} \
        --gres=gpu:$g -n$2 --ntasks-per-node=$g  ${SRUN_ARGS} \
        python -u $pyroot/models/imagenet/main.py --config ${cfg} \
        --save_path=algolib_gen/example/${MODEL_NAME} --port=$port \
        ${EXTRA_ARGS} \
        2>&1 | tee algolib_gen/example/${MODEL_NAME}/train_${MODEL_NAME}.log-$now
else
    OMPI_MCA_mpi_warn_on_fork=0 GLOG_vmodule=MemcachedClient=-1 \
    srun --mpi=pmi2 -p $partition --job-name=example_${MODEL_NAME} \
        --gres=gpu:$g -n$2 --ntasks-per-node=$g  ${SRUN_ARGS}  \
        python -u $pyroot/models/imagenet/main.py --config ${cfg} \
        --save_path=algolib_gen/example/${MODEL_NAME} --port=$port \
        ${EXTRA_ARGS} \
        2>&1 | tee algolib_gen/example/${MODEL_NAME}/train_${MODEL_NAME}.log-$now
fi
