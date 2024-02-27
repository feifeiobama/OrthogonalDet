#!/bin/bash

SETUP=${SETUP:-"10_10"}  # 10_10 or 15_5 or 19_1
PORT=${PORT:-"50210"}

python train_net.py --num-gpus 4 --dist-url tcp://127.0.0.1:${PORT} --task IOD/trainval --config-file configs/IOD/${SETUP}_0.yaml

python train_net.py --num-gpus 4 --dist-url tcp://127.0.0.1:${PORT} --task IOD/trainval --config-file configs/IOD/${SETUP}_1.yaml --resume MODEL.WEIGHTS output/IOD_${SETUP}/model_0017999.pth

python train_net.py --num-gpus 4 --dist-url tcp://127.0.0.1:${PORT} --task IOD/trainval --config-file configs/IOD/${SETUP}_ft.yaml --resume MODEL.WEIGHTS output/IOD_${SETUP}/model_0019999.pth