# We use 16 x 6 = 96 V100-32 GPUs
# On AiMOS cluster [https://docs.cci.rpi.edu/clusters/DCS_Supercomputer/]
# salloc --nodes 16 --time 6:00:00 --gres=gpu:32g:6 srun bash scripts/tgrt_question_generate_65b_base.sh
# We use the deduplicated topics from epoch 5 of TGRT topic brainstorming to generate topic-guided questions.

set -e
set -x

export PYTHONPATH="$PWD:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
export MODEL_DIR="/your/model/dir"
export DATA_DIR="/your/data/dir"
export OMP_NUM_THREADS=6
export GPUS_PER_NODE=6
export NUM_NODES=2
export MASTER_PORT=9901


LOCAL_NODE_RANK=$((SLURM_PROCID % NUM_NODES))
GROUP_RANK=$((SLURM_PROCID / NUM_NODES))
GROUP_SIZE=$((SLURM_NNODES / NUM_NODES))
SYNC_NODE_RANK=$((GROUP_RANK * NUM_NODES))

# MASTER_ADDR should be SYNC_NODE_RANK-th node in $(scontrol show hostnames $SLURM_JOB_NODELIST)
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n $((SYNC_NODE_RANK + 1)) | tail -n 1)

echo "$MASTER_ADDR, $GROUP_RANK / $GROUP_SIZE: $LOCAL_NODE_RANK"

torchrun --nproc_per_node $GPUS_PER_NODE \
  --nnodes $NUM_NODES \
  --node_rank $LOCAL_NODE_RANK \
  --master_addr $MASTER_ADDR \
  --master_port $MASTER_PORT \
  generate_tgrt_question.py \
  --ckpt_dir $MODEL_DIR/llama-65b-base-12shards \
  --tokenizer_path $MODEL_DIR/tokenizer.model \
  --max_shared_seq_len 256 \
  --max_seq_len 512 \
  --generate_max_len 384 \
  --group_rank $GROUP_RANK \
  --group_size $GROUP_SIZE \
  --meta_prompt_file "../../prompts/tgrt_self_instruct_question_generation_prompt.txt" \
  --seed_questions_path "../../prompts/tgrt_self_instruct_seed_questions.jsonl" \
  --seed_topics_path "$DATA_DIR/llama65b_tgrt_topics_epoch5_deduplicated.jsonl" \
  --output_path "$DATA_DIR/llama65b_tgrt_questions_${GROUP_SIZE}shards_${GROUP_RANK}.jsonl" \
  --request_batch_size 32 \
  --num_examples 5 \
  --num_instructions_to_generate 5 \
  --temperature 1.0 \
  --top_p 0.95
