#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)

# Distributed training configuration
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-$(shuf -i 20001-29999 -n 1)}
NPROC_PER_NODE=${NPROC_PER_NODE:-$(nvidia-smi --list-gpus | wc -l)}
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}

# DeepSpeed configuration
deepspeed=${DEEPSPEED_CONFIG:-${SCRIPT_DIR}/zero3.json}

# Model configuration
llm=${LLM_PATH:-/share/project/zhouenshen/hpfs/ckpt/vlm/Qwen3-VL-2B-Instruct}

# Training hyperparameters
lr=${LR:-2e-5}
active_mlp_lr=${ACTIVE_MLP_LR:-1e-3}
batch_size=${BATCH_SIZE:-14}
grad_accum_steps=${GRAD_ACCUM_STEPS:-4}
num_train_epochs=${NUM_TRAIN_EPOCHS:-1}
max_steps=${MAX_STEPS:--1}
image_max_pixels=${IMAGE_MAX_PIXELS:-262144}
image_min_pixels=${IMAGE_MIN_PIXELS:-1024}
video_max_pixels=${VIDEO_MAX_PIXELS:-262144}
video_min_pixels=${VIDEO_MIN_PIXELS:-1024}
model_max_length=${MODEL_MAX_LENGTH:-81920}
save_steps=${SAVE_STEPS:-1000}
save_total_limit=${SAVE_TOTAL_LIMIT:-1}
logging_steps=${LOGGING_STEPS:-1}
warmup_ratio=${WARMUP_RATIO:-0.03}
dataloader_num_workers=${DATALOADER_NUM_WORKERS:-4}

# ActiveQwen configuration
active_latent_token_count=${ACTIVE_LATENT_TOKEN_COUNT:-12}
active_prompt_length=${ACTIVE_PROMPT_LENGTH:-64}
active_projector_depth=${ACTIVE_PROJECTOR_DEPTH:-2}
active_target_dim=${ACTIVE_TARGET_DIM:-768}
active_projector_tunable=${TUNE_ACTIVE_PROJECTOR:-True}
active_ce_loss_weight=${ACTIVE_CE_LOSS_WEIGHT:-1.0}
active_3d_loss_weight=${ACTIVE_3D_LOSS_WEIGHT:-0.5}

# Training entry point
entry_file=${ENTRY_FILE:-${PROJECT_ROOT}/qwenvl/train/train_qwen.py}


dataset_names=(
    # hstar_sft_512x384_fov_90
    # hstar_sft_512x384_fov_90_latent

    raw_pano_512x384_fov_90_latent
    raw_pano_filter_512x384_fov_90

    deepeyes_train
    deepeyes_train_org
    deepeyes_rl
    deepeyes_rl_org
    deepeyes_rl_vstar
    deepeyes_rl_vstar_org
    visualprobe_train
    visualprobe_train_org
)
datasets=$(IFS=,; echo "${dataset_names[*]}")

# Output configuration
run_name=${RUN_NAME:-ActiveQwen3VL_2B_AP_PAP_HSTAR}
output_dir=${OUTPUT_DIR:-${PROJECT_ROOT}/output/activeqwen3vl_2b_visual_search}

# Training arguments
args="
    --deepspeed ${deepspeed} \
    --model_name_or_path ${llm} \
    --dataset_use ${datasets} \
    --data_flatten True \
    --tune_mm_vision False \
    --tune_mm_mlp False \
    --tune_mm_llm True \
    --tune_active_projector True \
    --activeqwen_enable True \
    --activeqwen_latent_token_count ${active_latent_token_count} \
    --activeqwen_projector_prompt_length ${active_prompt_length} \
    --activeqwen_projector_depth ${active_projector_depth} \
    --activeqwen_target_dim ${active_target_dim} \
    --active_ce_loss_weight ${active_ce_loss_weight} \
    --active_3d_loss_weight ${active_3d_loss_weight} \
    --bf16 \
    --output_dir ${output_dir} \
    --num_train_epochs ${num_train_epochs} \
    --per_device_train_batch_size ${batch_size} \
    --gradient_accumulation_steps ${grad_accum_steps} \
    --max_pixels ${image_max_pixels} \
    --min_pixels ${image_min_pixels} \
    --video_max_pixels ${video_max_pixels} \
    --video_min_pixels ${video_min_pixels} \
    --eval_strategy no \
    --save_strategy steps \
    --save_steps ${save_steps} \
    --save_total_limit ${save_total_limit} \
    --learning_rate ${lr} \
    --mm_projector_lr ${lr} \
    --active_projector_lr ${active_mlp_lr} \
    --weight_decay 0 \
    --warmup_ratio ${warmup_ratio} \
    --max_grad_norm 1 \
    --lr_scheduler_type cosine \
    --logging_steps ${logging_steps} \
    --model_max_length ${model_max_length} \
    --gradient_checkpointing True \
    --dataloader_num_workers ${dataloader_num_workers} \
    --run_name ${run_name} \
    --report_to none"

if [[ "${max_steps}" != "-1" ]]; then
    args="${args} --max_steps ${max_steps}"
fi

# Launch training
cd "${PROJECT_ROOT}"
torchrun --nnodes=${NNODES} \
         --node_rank=${NODE_RANK} \
         --nproc_per_node=${NPROC_PER_NODE} \
         --master_addr=${MASTER_ADDR} \
         --master_port=${MASTER_PORT} \
         ${entry_file} ${args}
