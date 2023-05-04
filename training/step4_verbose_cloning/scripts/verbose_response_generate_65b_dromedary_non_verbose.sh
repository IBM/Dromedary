# We use 64 x 6 = 384 V100-32GB GPUs
# On AiMOS cluster [https://docs.cci.rpi.edu/clusters/DCS_Supercomputer/]
# salloc --nodes 64 --time 6:00:00 --gres=gpu:32g:6 srun bash scripts/verbose_response_generate_65b_dromedary_non_verbose.sh

set -e
set -x

export PYTHONPATH="$PWD:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
export DATA_DIR="/gpfs/u/home/AICD/AICDsnzh/scratch/llama_data"
export OUTPUT_DIR="/gpfs/u/home/AICD/AICDsnzh/scratch/outputs"
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
  generate_verbose_response.py \
  --ckpt_dir $OUTPUT_DIR/dromedary-65b-lora-non-verbose-12shards \
  --tokenizer_path $OUTPUT_DIR/tokenizer.model \
  --generate_max_len 512 \
  --max_seq_len 512 \
  --max_shared_seq_len 640 \
  --max_batch_size 64 \
  --group_rank $GROUP_RANK \
  --group_size $GROUP_SIZE \
  --input_file "$DATA_DIR/dromedary65b_verbose_clone_input.json" \
  --output_file "$DATA_DIR/final/dromedary65b_verbose_clone_${GROUP_SIZE}shards_${GROUP_RANK}.json" \
  --meta_prompt_file "../../prompts/verbose_dromedary_prompt.txt" \
  --temperature 0.3 \
  --top_p 0.7 \
  --unitoken_frequency_penalty 0.0 \
  --bitoken_frequency_penalty 0.0 \
  --tritoken_frequency_penalty 1.0 \
  --quadtoken_frequency_penalty 2.0
