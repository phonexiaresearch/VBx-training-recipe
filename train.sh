# @Created Time:   2019-03-02
# @Author: Shuai Wang
# @Email: wsstriving@gmail.com
# @Last Modified Time: 2019-08-18

#!/bin/bash

if [ "$#" -ne 2 ]; then
  echo "Illegal number of parameters. Expecting egs directory and output directory."
  exit 1
fi

egs_dir=$1
out_dir=$2

AVAIL_GPUS=$(local/utils/free-gpus.sh)
ngpus=2

# linear | arc_margin | sphere |add_margin
metric="add_margin"
model="ResNet101"
embed_dim=256

export CUDA_VISIBLE_DEVICES=${AVAIL_GPUS:0:$((${ngpus}*2-1))}

# horovod stuff
export NCCL_SOCKET_IFNAME=bond0.6
export CUDA_LAUNCH_BLOCKING=1
export HOROVOD_GPU_ALLREDUCE=NCCL

. ./path.sh


# final model trained using 6x RTX 2080Ti, modify batchsize accordingly
horovodrun -np ${ngpus} --mpi --autotune \
    python local/train_pytorch_dnn.py --model ${model} \
        --num-targets 8178 \
        --dir ${out_dir}/${model}_${metric}_embed${embed_dim}_${ngpus}gpu \
        --metric ${metric} \
        --egs-dir ${egs_dir} \
        --minibatch-size 12 \
        --embed-dim ${embed_dim} \
        --warmup-epochs 0 \
        --initial-effective-lrate 0.01 \
        --final-effective-lrate 0.00005 \
        --initial-margin-m 0.05 \
        --final-margin-m 0.2 \
        --optimizer SGD \
        --momentum 0.9 \
        --optimizer-weight-decay 0.0001 \
        --preserve-model-interval 30 \
        --num-epochs 3 \
        --apply-cmn no \
        --fix-margin-m 2

