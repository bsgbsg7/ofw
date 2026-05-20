#!/bin/bash
# Full pipeline for OTFS variant: train from scratch + fine-tune
set -e

source activate deepofw
CONDA_PREFIX=$(conda info --base)/envs/deepofw
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cudnn/lib:$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cuda_runtime/lib:$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cublas/lib:$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cuda_nvrtc/lib:/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH

cd /home/v-haoliu3/EfficientLLM/ShiZheng/DeepOFW/Main_OTFS

echo "========== Stage 1: Training from scratch (TV channel) =========="
python train_stage1_otfs.py

echo "========== Stage 2: Fine-tuning =========="
python train_stage2_otfs.py

echo "========== Stage 3: Extended fine-tuning =========="
python train_stage3_otfs.py

echo "========== Done! =========="
