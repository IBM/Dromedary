# We use 1 x 6 = 6 V100-32GB GPUs
# On AiMOS cluster [https://docs.cci.rpi.edu/clusters/DCS_Supercomputer/]
# salloc --nodes 1 --time 6:00:00 --gres=gpu:32g:6 srun bash scripts/topic_generate_65b_base.sh
# This script will grow the topic pool for 9 epochs

set -e
set -x

export PYTHONPATH="$PWD:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
export MODEL_DIR="/your/model/dir"
export DATA_DIR="/your/data/dir"
export OMP_NUM_THREADS=6
export GPUS_PER_NODE=6
export NUM_NODES=1
export MASTER_PORT=9901


LOCAL_NODE_RANK=0
SYNC_NODE_RANK=0

# MASTER_ADDR should be SYNC_NODE_RANK-th node in $(scontrol show hostnames $SLURM_JOB_NODELIST)
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n $((SYNC_NODE_RANK + 1)) | tail -n 1)

torchrun --nproc_per_node $GPUS_PER_NODE \
  --nnodes $NUM_NODES \
  --node_rank $LOCAL_NODE_RANK \
  --master_addr $MASTER_ADDR \
  --master_port $MASTER_PORT \
  generate_tgrt_topic.py \
  --ckpt_dir $MODEL_DIR/llama-65b-base-6shards \
  --tokenizer_path $MODEL_DIR/tokenizer.model \
  --max_shared_seq_len 256 \
  --max_seq_len 128 \
  --generate_max_len 128 \
  --meta_prompt_file "../../prompts/prompts/tgrt_self_instruct_topic_brainstorm_prompt.txt" \
  --seed_questions_path "../../prompts/tgrt_self_instruct_seed_questions.jsonl" \
  --output_path "$DATA_DIR/llama65b_tgrt_topics.jsonl" \
  --num_examples 5 \
  --generation_epoch 9 \
  --request_batch_size 32 \
  --temperature 1.0 \
  --top_p 0.98
