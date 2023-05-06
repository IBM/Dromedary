#!/bin/bash
set -e
set -x

export PYTHONPATH="$PWD:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MODEL_DIR="/path/to/your/model/dir"
export OMP_NUM_THREADS=8
export GPUS_PER_NODE=8
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=9901
export TOTAL_NUM_GPUS=$(( $SLURM_NNODES * $GPUS_PER_NODE ))

N_SHARDS=8
CKPT_NAME="dromedary-65b-lora-final"

torchrun --nproc_per_node $GPUS_PER_NODE \
  --nnodes $SLURM_NNODES \
  --node_rank $SLURM_PROCID \
  --master_addr $MASTER_ADDR \
  --master_port $MASTER_PORT \
  run_stream_chatbot_demo.py \
  --ckpt_dir $MODEL_DIR/$CKPT_NAME-${N_SHARDS}shards \
  --tokenizer_path $MODEL_DIR/tokenizer.model \
  --max_seq_len 2048 \
  --max_batch_size 1 \
  --meta_prompt_file "../prompts/inference_prompts/dromedary_verbose_prompt.txt"
