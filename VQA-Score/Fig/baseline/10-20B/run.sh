#!/bin/bash

. /home/zhuyao/anaconda3/etc/profile.d/conda.sh



conda activate LLAVA
CUDA_VISIBLE_DEVICES=5 python /home/zhuyao/Sunpeng/wxx_data_eval/baseline/10-20B/LLAVA_13B_v1.6.py
conda deactivate


conda activate LLAVA
CUDA_VISIBLE_DEVICES=5 python /home/zhuyao/Sunpeng/wxx_data_eval/baseline/10-20B/LLAVA_13B.py
conda deactivate



