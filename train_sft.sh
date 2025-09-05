accelerate launch sft.py \
    --baseModel Qwen/Qwen2.5-7B-Instruct \
    --datasetPath AI-MO/NuminaMath-TIR \
    --outputDir /mnt/disk1/grpo_vlm/Qwen7B_NuminaMath \
    --seed 42 \
    --trainSamples 10000 \
    --perDeviceBatch 1 \
    --gradAccum 16 \
    --epochs 3 \
    --bf16