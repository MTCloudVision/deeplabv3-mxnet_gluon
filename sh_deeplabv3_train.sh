#!/usr/bin/env bash
export MXNET_CUDNN_AUTOTUNE_DEFAULT=0
export PYTHONUNBUFFERED=1
export MXNET_ENABLE_GPU_P2P=0
python3 -u train.py --train_epochs 83 --resume 45 --batch_size 8 --data_dir ./VOC2012/ --base_architecture resnet_v2_50 \
--pre_trained_model resnet_v2_50.params --output_stride 8 --freeze_batch_norm 1 --initial_learning_rate 1e-3 \
--weight_decay 2e-4 --gpus 0,1 --max_iter 30000 --aspp_or_vortex 1
