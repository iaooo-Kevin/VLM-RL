cd /mnt/disk1/grpo_vlm
#Reduce number of generations if OOM
accelerate launch grpo.py \
    --model_name_or_path Qwen/Qwen2.5-VL-3B-Instruct \
    --dataset_name  DaveKevin/GeoQA-GoldenCoT \
    --use_gspo true \
    --use_dapo false \
    --use_drgrpo false \
    --intermediate_reasoning false \
    --output_dir grpo_qwen_vlm \
    --use_peft \
    --lora_target_modules q_proj v_proj \
    --torch_dtype bfloat16 \
    --max_prompt_length 2048 \
    --learning_rate 1e-5 \
    --num_generations 4 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2