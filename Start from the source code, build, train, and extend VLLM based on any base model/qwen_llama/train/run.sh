
CUDA_VISIBLE_DEVICES=6,7 accelerate launch /home/zhuyao/Sunpeng/llava_qwen/train.py \
    --model_dir "/home/zhuyao/Sunpeng/llava_qwen/tes" \
    --processor_path "/home/zhuyao/Sunpeng/models/qwen_2B_instruct" \
    --output_dir "/home/zhuyao/Sunpeng/llava_qwen/result" \
    --min_image_tokens 5 \
    --max_image_tokens 10 \
    --train_data_path "/home/zhuyao/Sunpeng/finetune_qwen2vl/data/pokemen_train.json" \
    --eval_data_path "/home/zhuyao/Sunpeng/finetune_qwen2vl/data/pokemen_eval.json" \
    --num_train_epochs 3 \
    --learning_rate 1e-5 \
    --warmup_steps 0 \
    --lr_scheduler_type "cosine" \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --gpu_nums 2 \
    --log_dir "/home/zhuyao/Sunpeng/llava_qwen/log" \
    --log_steps 2\
    --save_steps 20\
    --system_message "You're a fan of animation."