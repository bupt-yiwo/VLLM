! export CUDA_VISIBLE_DEVICES=0,1,2,3

torchrun --nproc-per-node=4 /home/zhuyao/Sunpeng/llava_qwen/SP/pretrain_instruct_downstream/train_trainer.py \
  --model_dir "/home/zhuyao/Sunpeng/llava_qwen/tes" \
  --tokenizer_path "/home/zhuyao/Sunpeng/llava_qwen/tes" \
  --processor_path "/home/zhuyao/Sunpeng/models/qwen_2B_instruct" \
  --train_data_path "/home/zhuyao/Sunpeng/llava_qwen/data/pretrain/blip_laion_cc_sbu_558k_reconstructed.json" \
  --output_dir "/home/zhuyao/Sunpeng/llava_qwen/check_point/pretrain" \
  --deepspeed_config "/home/zhuyao/Sunpeng/A100_QWEN2VL/zero_stage2.json" \
  --finetune_choice "full_finetune_merger"\
  --lora_modules "q_proj", "k_proj", "v_proj", "o_proj" \
  --lora_rank 4  \
  --lora_alpha 8 \
  --max_image_tokens 336 \
  --min_image_tokens 4 \
  --per_device_train_batch_size 2 \
  --per_device_eval_batch_size 2 \
  --num_train_epochs 1 \
  --logging_steps 10 \
  --save_steps 545 \
  --save_total_limit 100 \
  --gradient_accumulation_steps 16 \
  --learning_rate 1e-3 \
  --warmup_ratio 0.03 \
  --weight_decay 0.0 \
  --lr_scheduler_type "cosine" \
  --system_message "You are a helpful assistant."\
  --eval_data_path "/home/zhuyao/Sunpeng/finetune_qwen2vl/data/pokemen_eval.json" \
  --evaluation_strategy "no" \
  --eval_steps 1\
  --gradient_checkpointing False


