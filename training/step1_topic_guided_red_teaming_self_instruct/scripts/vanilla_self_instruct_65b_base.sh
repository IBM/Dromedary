# On AiMOS cluster [https://docs.cci.rpi.edu/clusters/DCS_Supercomputer/]
# salloc --nodes 64 --time 6:00:00 --gres=gpu:32g:6 srun bash scripts/vanilla_self_instruct_65b_base.sh
# This would results in (64 / 2) * 8192 = 262144 instructions generated

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

torchrun --nproc_per_node $GPUS_PER_NODE \
  --nnodes $NUM_NODES \
  --node_rank $LOCAL_NODE_RANK \
  --master_addr $MASTER_ADDR \
  --master_port $MASTER_PORT \
  generate_vanilla_instruction.py \
  --ckpt_dir $MODEL_DIR/llama-65b-base-12shards \
  --tokenizer_path $MODEL_DIR/tokenizer.model \
  --max_shared_seq_len 512 \
  --max_seq_len 1024 \
  --generate_max_len 1024 \
  --group_rank $GROUP_RANK \
  --group_size $GROUP_SIZE \
  --meta_prompt_file "../prompts/self_instruct_prompt.txt" \
  --seed_tasks_path "../prompts/self_instruct_seed_tasks.jsonl" \
  --output_path "$DATA_DIR/llama65b_self_instruct_${GROUP_SIZE}shards_${GROUP_RANK}.jsonl" \
  --num_instructions_to_generate 8192 \
  --num_prompt_instructions 5 \
  --request_batch_size 64 \
  --temperature 1.0 \
  --top_p 0.98 \
  --seed 42
