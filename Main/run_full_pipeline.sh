#!/bin/bash
# Full pipeline: train qQ model from scratch (2 stages) + evaluate + generate waveform figures
set -e

# Activate environment and set GPU libs
source activate deepofw
CONDA_PREFIX=$(conda info --base)/envs/deepofw
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cudnn/lib:$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cuda_runtime/lib:$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cublas/lib:$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cuda_nvrtc/lib:/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH

cd /home/v-haoliu3/EfficientLLM/ShiZheng/DeepOFW/Main

echo "========== Stage 1: Training from scratch =========="
python train_stage1.py

echo "========== Stage 2: Fine-tuning =========="
python train_stage2.py

echo "========== Evaluation: Waveform scan =========="
python -c "
import sys
sys.path.append('src')
from src.qQ_Method.qQ_waveforms_inspec_scan import *
"

echo "========== Done! =========="
