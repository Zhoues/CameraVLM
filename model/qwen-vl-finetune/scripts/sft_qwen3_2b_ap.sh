#!/bin/bash

# Distributed training configuration
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-$(shuf -i 20001-29999 -n 1)}
NPROC_PER_NODE=$(nvidia-smi --list-gpus | wc -l)  # Automatically detects available GPUs
NNODES=${WORLD_SIZE:-1}

# DeepSpeed configuration
deepspeed=./scripts/zero3.json

# Model configuration
llm=/share/project/zhouenshen/hpfs/ckpt/vlm/Qwen3-VL-2B-Instruct

# Training hyperparameters aligned with LlamaFactory/examples/train_full/qwen3vl_full_sft_ap_2B_filter.yaml
lr=2e-5
batch_size=14
grad_accum_steps=4
num_train_epochs=1
image_max_pixels=262144
image_min_pixels=1024
video_max_pixels=262144
video_min_pixels=1024
model_max_length=81920
save_steps=3000
logging_steps=1
warmup_ratio=0.03

# Training entry point
entry_file=qwenvl/train/train_qwen.py

dataset_names=(
    ca1m_referring_512x384_fov_90
    ca1m_vacant_512x384_fov_90

    pap_512x384_fov_90
    pap_multi_step_512x384_fov_90
    pap_filter_512x384_fov_90
    pap_retrieval_512x384_fov_90

    raw_pano_512x384_fov_90
    raw_pano_multi_step_512x384_fov_90
    raw_pano_filter_512x384_fov_90
    raw_pano_retrieval_512x384_fov_90

    # hstar_bench_512x384_fov_90
    hstar_sft_512x384_fov_90
    hstar_sft_ours_512x384_fov_90
    hstar_sft_ours_filter_512x384_fov_90
    hstar_sft_ours_retrieval_512x384_fov_90
    # vsi_590k
    # libero_fast_vlm
)
datasets=$(IFS=,; echo "${dataset_names[*]}")

# Output configuration
run_name="qwen3vl_2b_ap_all_data"
output_dir=./output/qwen3vl_2b_ap_all_data

# Training arguments
args="
    --deepspeed ${deepspeed} \
    --model_name_or_path "${llm}" \
    --dataset_use ${datasets} \
    --data_flatten True \
    --tune_mm_vision False \
    --tune_mm_mlp True \
    --tune_mm_llm True \
    --bf16 \
    --output_dir ${output_dir} \
    --num_train_epochs ${num_train_epochs} \
    --per_device_train_batch_size ${batch_size} \
    --gradient_accumulation_steps ${grad_accum_steps} \
    --max_pixels ${image_max_pixels} \
    --min_pixels ${image_min_pixels} \
    --video_max_pixels ${video_max_pixels} \
    --video_min_pixels ${video_min_pixels} \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps ${save_steps} \
    --save_total_limit 1 \
    --learning_rate ${lr} \
    --weight_decay 0 \
    --warmup_ratio ${warmup_ratio} \
    --max_grad_norm 1 \
    --lr_scheduler_type "cosine" \
    --logging_steps ${logging_steps} \
    --model_max_length ${model_max_length} \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --run_name ${run_name} \
    --report_to none"

# Launch training
torchrun --nproc_per_node=${NPROC_PER_NODE} \
         --master_addr=${MASTER_ADDR} \
         --master_port=${MASTER_PORT} \
         ${entry_file} ${args}
