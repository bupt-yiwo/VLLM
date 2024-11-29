! export CUDA_VISIBLE_DEVICES=4,5,6,7

torchrun --nproc-per-node=4 /home/zhuyao/Sunpeng/llava_qwen/train_trainer.py \
  --model_dir "/home/zhuyao/Sunpeng/llava_qwen/check_point/instruct_525k" \
  --tokenizer_path "/home/zhuyao/Sunpeng/llava_qwen/tes" \
  --processor_path "/home/zhuyao/Sunpeng/models/qwen_2B_instruct" \
  --train_data_path "/home/zhuyao/Sunpeng/llava_qwen/data/cot/train_split.json" \
  --output_dir "/home/zhuyao/Sunpeng/llava_qwen/check_point/downstream_tasks/cot" \
  --deepspeed_config "/home/zhuyao/Sunpeng/llava_qwen/zero_stage2.json" \
  --finetune_choice "full_finetune_llm"\
  --lora_modules "q_proj", "k_proj", "v_proj", "o_proj" \
  --lora_rank 8  \
  --lora_alpha 16 \
  --max_image_tokens 336 \
  --min_image_tokens 4 \
  --per_device_train_batch_size 2 \
  --per_device_eval_batch_size 2 \
  --num_train_epochs 2 \
  --logging_steps 5 \
  --save_steps 445 \
  --save_total_limit 100 \
  --gradient_accumulation_steps 8 \
  --learning_rate 1e-5 \
  --warmup_ratio 0.03 \
  --weight_decay 0.0 \
  --lr_scheduler_type "cosine" \
  --system_message "You are a helpful assistant."\
  --eval_data_path "/home/zhuyao/Sunpeng/llava_qwen/data/cot/val_split.json" \
  --evaluation_strategy "steps" \
  --eval_steps 445\
  --gradient_checkpointing False


