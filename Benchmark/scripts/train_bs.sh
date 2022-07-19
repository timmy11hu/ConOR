#!/bin/bash

python -m torch.distributed.launch --nproc_per_node=4 \
../train_test/train_bs.py \
--config config/dorn_kitti.yaml \
--loss_type conor \
--mode wild \
--epoch 2 \
--BS_num 20
