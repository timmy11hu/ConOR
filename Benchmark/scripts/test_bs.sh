#!/bin/bash

python ../train_test/test_bs.py \
--config config/dorn_kitti.yaml \
--loss_type conor \
--mode multiplier \
--BS_num 20
