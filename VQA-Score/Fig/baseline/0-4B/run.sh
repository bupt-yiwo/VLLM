#!/bin/bash

. /home/zhuyao/anaconda3/etc/profile.d/conda.sh

# 执行 MiniCPM_V_2.py 脚本
conda activate MINICPM
CUDA_VISIBLE_DEVICES=4 python /home/zhuyao/Sunpeng/wxx_data_eval/baseline/0-4B/MiniCPM_V_2.py
conda deactivate

# 执行 deepseek1.3B.py 脚本
conda activate deepseek
CUDA_VISIBLE_DEVICES=4 python /home/zhuyao/Sunpeng/wxx_data_eval/baseline/0-4B/deepseek1.3B.py
conda deactivate

# 执行 InternVL2_2B.py 脚本
conda activate intervl
CUDA_VISIBLE_DEVICES=4 python /home/zhuyao/Sunpeng/wxx_data_eval/baseline/0-4B/InternVL2_2B.py
conda deactivate


# 执行 LLaVA_OneVision_0.5B.py 脚本
conda activate llava_oneversion
CUDA_VISIBLE_DEVICES=4 python /home/zhuyao/Sunpeng/wxx_data_eval/baseline/0-4B/LLaVA_OneVision_0.5B.py
conda deactivate

# 执行 qwen2vl_2B.py 脚本
conda activate qwen2vl
CUDA_VISIBLE_DEVICES=4 python /home/zhuyao/Sunpeng/wxx_data_eval/baseline/0-4B/qwen2vl_2B.py
conda deactivate
