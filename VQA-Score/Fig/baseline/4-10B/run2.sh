#!/bin/bash

. /home/zhuyao/anaconda3/etc/profile.d/conda.sh


conda activate MINICPM
CUDA_VISIBLE_DEVICES=2,3,4 python /home/zhuyao/Sunpeng/wxx_data_eval/baseline/4-10B/MINICPM_8B.py
conda deactivate


conda activate LLAVA
CUDA_VISIBLE_DEVICES=3 python /home/zhuyao/Sunpeng/wxx_data_eval/baseline/4-10B/LLAVA_7B_v1.6.py
conda deactivate


conda activate LLAVA
CUDA_VISIBLE_DEVICES=3 python /home/zhuyao/Sunpeng/wxx_data_eval/baseline/4-10B/LLAVA_7B.py
conda deactivate



