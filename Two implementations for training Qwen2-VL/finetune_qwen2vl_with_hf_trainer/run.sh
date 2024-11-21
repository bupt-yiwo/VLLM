! export CUDA_VISIBLE_DEVICES=6,7

torchrun --nproc-per-node=2 /home/zhuyao/Sunpeng/finetune_qwen2vl/train_qwen2vl.py \
  --model_dir "/home/zhuyao/Sunpeng/models/qwen_2B_instruct" \
  --train_data_path "/home/zhuyao/Sunpeng/finetune_qwen2vl/data/pokemen_train.json" \
  --output_dir "/home/zhuyao/Sunpeng/finetune_qwen2vl/result" \
  --deepspeed_config "/home/zhuyao/Sunpeng/A100_QWEN2VL/zero_stage2.json" \
  --finetune_choice "full_finetune_llm"\
  --lora_modules "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj" \
  --lora_rank 4  \
  --lora_alpha 8 \
  --max_image_tokens 336 \
  --min_image_tokens 336 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --num_train_epochs 1 \
  --logging_steps 5 \
  --save_steps 250 \
  --save_total_limit 100 \
  --gradient_accumulation_steps 2 \
  --learning_rate 1e-5 \
  --warmup_steps 20 \
  --gradient_checkpointing False \
  --lr_scheduler_type "constant" \
  --system_message "You are a helpful assistant."\
  --eval_data_path "/home/zhuyao/Sunpeng/finetune_qwen2vl/data/pokemen_eval.json" \
  --evaluation_strategy "steps" \
  --eval_steps 10



