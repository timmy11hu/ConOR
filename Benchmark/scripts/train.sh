#!/bin/bash

python -m torch.distributed.launch --nproc_per_node=4 \
../train_test/train.py \
--config config/dorn_kitti.yaml \
--seed 0 \
--loss_type conor \
--epoch 10

