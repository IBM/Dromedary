# Multiple Choice Evaluation of Dromedary

## Prerequisites

Follow our [inference guide](https://github.com/IBM/Dromedary/tree/main/inference) to see how to deploy Dromedary/LLaMA on your own machine with [model parallel](https://github.com/facebookresearch/fairscale/tree/main/fairscale/nn/model_parallel) (which should be significantly faster than Hugging Face's default pipeline parallel when using multiple GPUs).

Generally, since Dromedary is a 65B model, it requires a minimum of 130GB GPU memory to accommodate the entirety of its model weights within the GPU memory.

## HHH Eval

```bash
git clone git@github.com:google/BIG-bench.git /your/path/to/bigbench/repo
```

```bash
#!/bin/bash
set -e
set -x

export PYTHONPATH="$PWD:$PYTHONPATH"
export OUTPUT_DIR="/path/to/your/model/storage"
export OMP_NUM_THREADS=/YOUR_NUM_GPUS
export GPUS_PER_NODE=/YOUR_NUM_GPUS
export NUM_NODES=1

export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n $((SYNC_NODE_RANK + 1)) | tail -n 1)
export MASTER_PORT=9901

CKPT_NAME=/your/sharded_ckpt
BIG_BENCH_HOME=/your/path/to/bigbench/repo

torchrun --nproc_per_node $GPUS_PER_NODE \
  --nnodes $NUM_NODES \
  --node_rank 0 \
  --master_addr $MASTER_ADDR \
  --master_port $MASTER_PORT \
  evaluate_hhh_eval.py \
  --ckpt_dir $OUTPUT_DIR/$CKPT_NAME \
  --tokenizer_path $OUTPUT_DIR/tokenizer.model \
  --big_bench_home $BIG_BENCH_HOME \
  --max_seq_len 1536 \
  --max_shared_seq_len 512 \
  --max_batch_size 4 \
  --group_rank $GROUP_RANK \
  --group_size $GROUP_SIZE \
  --meta_prompt_file "../prompts/inference_prompts/dromedary_verbose_prompt.txt"
```

## TruthfulQA MC

```bash
#!/bin/bash
set -e
set -x

export PYTHONPATH="$PWD:$PYTHONPATH"
export OUTPUT_DIR="/path/to/your/model/storage"
export OMP_NUM_THREADS=/YOUR_NUM_GPUS
export GPUS_PER_NODE=/YOUR_NUM_GPUS
export NUM_NODES=1

export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n $((SYNC_NODE_RANK + 1)) | tail -n 1)
export MASTER_PORT=9901

CKPT_NAME=/your/sharded_ckpt

torchrun --nproc_per_node $GPUS_PER_NODE \
  --nnodes $NUM_NODES \
  --node_rank 0 \
  --master_addr $MASTER_ADDR \
  --master_port $MASTER_PORT \
  evaluate_truthfulqa_mc.py \
  --ckpt_dir $OUTPUT_DIR/$CKPT_NAME \
  --tokenizer_path $OUTPUT_DIR/tokenizer.model \
  --max_seq_len 384 \
  --max_shared_seq_len 512 \
  --max_batch_size 32 \
  --group_rank $GROUP_RANK \
  --group_size $GROUP_SIZE \
  --meta_prompt_file "../prompts/inference_prompts/dromedary_verbose_prompt.txt"
```
