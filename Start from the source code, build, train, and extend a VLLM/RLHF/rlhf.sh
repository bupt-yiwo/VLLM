python /home/zhuyao/Sunpeng/llava_qwen/RLHF_dpo.py \
  --model_path "/home/zhuyao/Sunpeng/llava_qwen/check_point/instruct_525k" \
  --tokenizer_path "/home/zhuyao/Sunpeng/llava_qwen/tes" \
  --processor_path "/home/zhuyao/Sunpeng/models/qwen_2B_instruct" \
  --train_data_path "/home/zhuyao/Sunpeng/llava_qwen/data/rlhf/RLHF-V" \
  --output_dir "/home/zhuyao/Sunpeng/llava_qwen/check_point/DPO_5.73k" \
  --max_image_tokens 336 \
  --min_image_tokens 4 \
  --per_device_train_batch_size 2 \
  --num_train_epochs 3 \
  --log_steps 5 \
  --save_steps 150 \
  --gradient_accumulation_steps 32 \
  --learning_rate 1e-5 \



