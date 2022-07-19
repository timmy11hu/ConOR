#!/bin/bash

python ../train_test/test.py \
--config config/dorn_kitti.yaml \
--loss_type conor \
--mode SG \
--MC_num 50
