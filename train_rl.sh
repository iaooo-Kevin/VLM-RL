cd /mnt/disk1/grpo_vlm
#Reduce number of generations if OOM
#Max Prompt Length != None will lead to truncation errors; Ergo, -1 is advised so as it defaults to None.
accelerate launch grpo.py \
    --model_name_or_path Qwen/Qwen2.5-VL-3B-Instruct \
    --dataset_name  phronetic-ai/MedVMCQA \
    --use_gspo false \
    --use_dapo false \
    --use_drgrpo true \
    --use_mcq_reward true \
    --intermediate_reasoning false \
    --use_llm_judge false \
    --use_medical_reward false \
    --output_dir vrznmed_highrank \
    --use_peft \
    --attn_implementation flash_attention_2 \
    --lora_target_modules q_proj v_proj k_proj o_proj gate_proj up_proj down_proj \
    --torch_dtype bfloat16 \
    --lora_rank 264 \
    --learning_rate 1e-5 \
    --max_prompt_length -1 \
    --num_train_epochs 2 \
    --num_generations 4 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4
