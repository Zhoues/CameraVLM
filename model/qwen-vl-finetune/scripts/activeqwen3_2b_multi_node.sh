#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-/share/project/lmz/miniconda3/envs/llamafactory/bin/python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-/share/project/lmz/miniconda3/envs/llamafactory/bin/torchrun}"

cd "${PROJECT_ROOT}"

# Run name
run_name="activeqwen3vl_2b_2_nodes_gas_4_camera+visual_search+general"
# Output configuration
output_dir=${PROJECT_ROOT}/output/${run_name}
mkdir -p ${output_dir}
# Log path
log_path=${output_dir}/train_node_${RANK}.log
# DeepSpeed configuration
deepspeed=${PROJECT_ROOT}/scripts/zero3.json
# Model configuration
llm=/share/project/zhouenshen/hpfs/ckpt/vlm/Qwen3-VL-2B-Instruct
#################################################################
## NCCL
#################################################################
export GLOO_SOCKET_IFNAME=eth0
export NCCL_SOCKET_IFNAME=eth0
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=7

export ACCELERATE_CPU_AFFINITY=1
export NCCL_IB_HCA=mlx5_100,mlx5_101,mlx5_102,mlx5_103,mlx5_104,mlx5_105,mlx5_106,mlx5_107

# used for check save when communication
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=1000  # timeout set to 1 hour (unit: seconds)
export NCCL_SOCKET_TIMEOUT_MS=360000

# export CUDA_LAUNCH_BLOCKING=1
# export TORCH_DISTRIBUTED_DEBUG=DETAIL
# export NCCL_DEBUG=INFO
#################################################################

#################################################################
## ACCELERATE CONFIG (Distributed training configuration)
#################################################################
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NPROC_PER_NODE=8
export TOTAL_GPUS=$((NPROC_PER_NODE * WORLD_SIZE))
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-$(shuf -i 20001-29999 -n 1)}
NNODES=${WORLD_SIZE:-1}

# Training hyperparameters
lr=2e-5
batch_size=13
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

# ActiveQwen configuration
active_mlp_lr=${ACTIVE_MLP_LR:-1e-3}
active_latent_token_count=${ACTIVE_LATENT_TOKEN_COUNT:-12}
active_prompt_length=${ACTIVE_PROMPT_LENGTH:-64}
active_projector_depth=${ACTIVE_PROJECTOR_DEPTH:-2}
active_target_dim=${ACTIVE_TARGET_DIM:-768}
active_projector_tunable=${TUNE_ACTIVE_PROJECTOR:-True}
active_ce_loss_weight=${ACTIVE_CE_LOSS_WEIGHT:-1.0}
active_3d_loss_weight=${ACTIVE_3D_LOSS_WEIGHT:-0.5}

dataset_names=(
    ca1m_referring_512x384_fov_90
    ca1m_vacant_512x384_fov_90

    pap_512x384_fov_90
    pap_512x384_fov_90_latent
    pap_multi_step_512x384_fov_90
    pap_filter_512x384_fov_90
    pap_filter_vqa_512x384_fov_90
    pap_retrieval_512x384_fov_90

    raw_pano_512x384_fov_90
    raw_pano_512x384_fov_90_latent
    raw_pano_multi_step_512x384_fov_90
    raw_pano_filter_512x384_fov_90
    raw_pano_filter_vqa_512x384_fov_90
    raw_pano_retrieval_512x384_fov_90

    # hstar_bench_512x384_fov_90
    hstar_sft_512x384_fov_90
    hstar_sft_512x384_fov_90
    hstar_sft_512x384_fov_90_latent
    hstar_sft_512x384_fov_90_latent

    hstar_sft_ours_512x384_fov_90
    hstar_sft_ours_512x384_fov_90
    hstar_sft_ours_512x384_fov_90_latent
    hstar_sft_ours_512x384_fov_90_latent
    hstar_sft_ours_filter_512x384_fov_90
    hstar_sft_ours_filter_vqa_512x384_fov_90
    hstar_sft_ours_retrieval_512x384_fov_90

    deepeyes_train
    deepeyes_train_org
    deepeyes_rl
    deepeyes_rl_org

    deepeyes_rl_vstar
    deepeyes_rl_vstar_org
    deepeyes_rl_vstar
    deepeyes_rl_vstar_org
    deepeyes_rl_vstar
    deepeyes_rl_vstar_org
    visualprobe_train
    
    llava_onevision_1_5
)
datasets=$(IFS=,; echo "${dataset_names[*]}")

# Training arguments
# Launch training
${TORCHRUN_BIN} --nproc_per_node=${NPROC_PER_NODE} \
                --master_addr=${MASTER_ADDR} \
                --master_port=${MASTER_PORT} \
                qwenvl/train/train_qwen.py \
                --deepspeed ${deepspeed} \
                --model_name_or_path "${llm}" \
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
                --eval_strategy "no" \
                --save_strategy "steps" \
                --save_steps ${save_steps} \
                --save_total_limit 1 \
                --learning_rate ${lr} \
                --mm_projector_lr ${lr} \
                --active_projector_lr ${active_mlp_lr} \
                --weight_decay 0 \
                --warmup_ratio ${warmup_ratio} \
                --max_grad_norm 1 \
                --lr_scheduler_type "cosine" \
                --logging_steps ${logging_steps} \
                --model_max_length ${model_max_length} \
                --gradient_checkpointing True \
                --dataloader_num_workers 4 \
                --run_name ${run_name} \
                --report_to none \
                2>&1 | tee "${log_path}"
