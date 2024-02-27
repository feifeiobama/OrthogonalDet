#!/bin/bash

BENCHMARK=${BENCHMARK:-"M-OWODB"}  # M-OWODB or S-OWODB
PORT=${PORT:-"50210"}

# if raise error, change num_gpus to 1
if [ $BENCHMARK == "M-OWODB" ]; then
  python train_net.py --num-gpus 4 --dist-url tcp://127.0.0.1:${PORT} --task ${BENCHMARK}/t1 --config-file configs/${BENCHMARK}/t1.yaml --eval-only MODEL.WEIGHTS output/${BENCHMARK}/model_0019999.pth

  python train_net.py --num-gpus 4 --dist-url tcp://127.0.0.1:${PORT} --task ${BENCHMARK}/t2_ft --config-file configs/${BENCHMARK}/t2_ft.yaml --eval-only MODEL.WEIGHTS output/${BENCHMARK}/model_0049999.pth

  python train_net.py --num-gpus 4 --dist-url tcp://127.0.0.1:${PORT} --task ${BENCHMARK}/t3_ft --config-file configs/${BENCHMARK}/t3_ft.yaml --eval-only MODEL.WEIGHTS output/${BENCHMARK}/model_0079999.pth

  python train_net.py --num-gpus 4 --dist-url tcp://127.0.0.1:${PORT} --task ${BENCHMARK}/t4_ft --config-file configs/${BENCHMARK}/t4_ft.yaml --eval-only MODEL.WEIGHTS output/${BENCHMARK}/model_0109999.pth
else
  python train_net.py --num-gpus 4 --dist-url tcp://127.0.0.1:${PORT} --task ${BENCHMARK}/t1 --config-file configs/${BENCHMARK}/t1.yaml --eval-only MODEL.WEIGHTS output/${BENCHMARK}/model_0039999.pth

  python train_net.py --num-gpus 4 --dist-url tcp://127.0.0.1:${PORT} --task ${BENCHMARK}/t2_ft --config-file configs/${BENCHMARK}/t2_ft.yaml --eval-only MODEL.WEIGHTS output/${BENCHMARK}/model_0069999.pth

  python train_net.py --num-gpus 4 --dist-url tcp://127.0.0.1:${PORT} --task ${BENCHMARK}/t3_ft --config-file configs/${BENCHMARK}/t3_ft.yaml --eval-only MODEL.WEIGHTS output/${BENCHMARK}/model_0099999.pth

  python train_net.py --num-gpus 4 --dist-url tcp://127.0.0.1:${PORT} --task ${BENCHMARK}/t4_ft --config-file configs/${BENCHMARK}/t4_ft.yaml --eval-only MODEL.WEIGHTS output/${BENCHMARK}/model_0129999.pth
fi