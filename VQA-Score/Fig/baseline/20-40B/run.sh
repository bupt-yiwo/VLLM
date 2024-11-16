#!/bin/bash

. /home/zhuyao/anaconda3/etc/profile.d/conda.sh



conda activate LLAVA
CUDA_VISIBLE_DEVICES=6,7 python /home/zhuyao/Sunpeng/wxx_data_eval/baseline/20-40B/LLAVA_34B.py
conda deactivate


conda activate intervl
CUDA_VISIBLE_DEVICES=6,7 python /home/zhuyao/Sunpeng/wxx_data_eval/baseline/20-40B/internvl2_26B.py
conda deactivate

