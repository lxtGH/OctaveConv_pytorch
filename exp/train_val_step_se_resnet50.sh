#!/usr/bin/env bash

python -m torch.distributed.launch --nproc_per_node=8 main_imagenet.py \
-a se_resnet50 --data /data/lxt/ImageNet \
--epochs 120 \
--schedule 30 60 90 \
--wd 1e-4 --gamma 0.1 \
--train-batch 64 \
--pretrained False \
--pretrained_dir /home/user/pretrained \
-c checkpoints/imagenet/se_res50_bs_512 \
--opt-level O0 \
--wd-all \
--label-smoothing 0. \
--warmup_epochs 5