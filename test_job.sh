#!/bin/bash
#SBATCH -N 1
#SBATCH -p gpu_shared
#SBATCH -c 3
#SBATCH -t 00:10:00

module load pre2019
module load Python/3.6.1-intel-2016b
module load CUDA/10.0.130
module load cuDNN/7.4.2-CUDA-10.0.130

pip install --upgrade --user  pip
pip install --user torch
pip install --user torchvision

cp $HOME/imagenet/* "$TMPDIR"

python3 main.py -a resnet6 -model resnet6 --epochs 1 --dist-url 'tcp://127.0.0.1:FREEPORT' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 /storage/Cars

cp "$TMPDIR"/imagenet/log/* $HOME/imagenet/log
cp "$TMPDIR"/imagenet/model/* $HOME/imagenet/model

cp $HOME/imagenet/scripts/main.py "$TMPDIR"

# python3 main.py -a resnet6 -model resnet6 --epochs 1 /storage/Cars
