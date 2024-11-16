#!/bin/bash

. /home/zhuyao/anaconda3/etc/profile.d/conda.sh


conda activate deepseek
CUDA_VISIBLE_DEVICES=2 python /home/zhuyao/Sunpeng/wxx_data_eval/baseline/4-10B/GLM4V_9B.py
conda deactivate


conda activate qwen2vl
CUDA_VISIBLE_DEVICES=2 python /home/zhuyao/Sunpeng/wxx_data_eval/baseline/4-10B/qwen2vl_7B.py
conda deactivate


conda activate intervl
CUDA_VISIBLE_DEVICES=2 python /home/zhuyao/Sunpeng/wxx_data_eval/baseline/4-10B/internvl2_8B.py
conda deactivate


