
CUDA_VISIBLE_DEVICES=6,7 accelerate launch /home/zhuyao/Sunpeng/finteune_qwen2vl_no_trainer/train.py \
    --model_dir "/home/zhuyao/Sunpeng/models/qwen_2B_instruct" \
    --output_dir "/home/zhuyao/Sunpeng/finteune_qwen2vl_no_trainer/result" \
    --min_resolution 336 \
    --max_resolution 336 \
    --train_data_path "/home/zhuyao/Sunpeng/finetune_qwen2vl/data/pokemen_train.json" \
    --eval_data_path "/home/zhuyao/Sunpeng/finetune_qwen2vl/data/pokemen_eval.json" \
    --num_train_epochs 3 \
    --learning_rate 1e-5 \
    --warmup_steps 0 \
    --lr_scheduler_type "cosine" \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 1 \
    --gpu_nums 2 \
    --log_dir "/home/zhuyao/Sunpeng/finteune_qwen2vl_no_trainer/log" \
    --log_steps 2\
    --save_steps 2\
    --system_message "You're a fan of animation."
