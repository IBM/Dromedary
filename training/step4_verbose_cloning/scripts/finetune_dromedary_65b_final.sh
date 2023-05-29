# We use 16 x 6 = 96 V100-32GB GPUs
# On AiMOS cluster [https://docs.cci.rpi.edu/clusters/DCS_Supercomputer/]
# salloc --nodes 16 --time 6:00:00 --gres=gpu:32g:6 srun bash scripts/finetune_dromedary_65b_final.sh

# Due to some unknown issues in HF datasets library, we recommend run `finetune.py`
# with --fake_run flag to prepare the dataset on your local machine,
# and then submit the slurm training job to the cluster.
set -e
set -x

cd ..

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
export MODEL_DIR="/your/model/dir"
export DATA_DIR="/your/data/dir"
export PYTHONPATH="$PWD:$PYTHONPATH"
export GPUS_PER_NODE=6
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=9901
export TOTAL_NUM_GPUS=$(( $SLURM_NNODES * $GPUS_PER_NODE ))

verbose_value=$(($SLURM_PROCID == 0))

if [ $verbose_value -eq 1 ]; then
    verbose_output=""
else
    verbose_output="--disable_verbose True"
fi

TOTAL_BATCH_SIZE=768
LEARNING_RATE=4e-4
NUM_EPOCHS=1
CKPT_STEPS=50

MICRO_BATCH_SIZE=16
GRADIENT_ACCUMULATION_STEPS=$(($TOTAL_BATCH_SIZE / $MICRO_BATCH_SIZE / $TOTAL_NUM_GPUS))

accelerate launch \
    --num_processes=$TOTAL_NUM_GPUS --num_machines=$SLURM_NNODES --machine_rank=$SLURM_PROCID \
    --main_process_ip $MASTER_ADDR --main_process_port $MASTER_PORT \
    --deepspeed_multinode_launcher "standard" \
    finetune.py \
    --num_warmup_steps 100 \
    --batch_size $TOTAL_BATCH_SIZE \
    --micro_batch_size $MICRO_BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --num_epochs $NUM_EPOCHS \
    --ds_gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --base_model "/path/to/your/llama-65b-hf" \
    --output_dir "$MODEL_DIR/dromedary-65b-lora-final" \
    --run_tensorboard_dir True \
    --checkpointing_steps $CKPT_STEPS \
    --resume_from_checkpoint True \
    --data_path "$DATA_DIR/llama65b_verbose_clone_merged.json" \
    --meta_prompt_pattern "../prompts/inference_prompts/dromedary_*prompt_distill.txt" \
    --add_eos_token False \
    --cutoff_len 768 \
    --train_on_inputs False \
    --lora_target_modules='[q_proj,k_proj,v_proj,o_proj]' \
    --lora_r=16 \
    $verbose_output
