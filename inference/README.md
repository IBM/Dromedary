# Chatbot Demo

## Quick Start

Assuming you have 2 A100-80GB GPUs and have devide the Dromedary/LLaMA checkpoints into 2 shards.
```bash
bash scripts/demo_dromedary_2shards.sh
```

Or assuming you have 6 V100-32GB GPUs and have devide the Dromedary/LLaMA checkpoints into 6 shards.
```bash
bash scripts/demo_dromedary_6shards.sh
```

## Further Customization

Generally, since the Dromedary is a 65B model, it requires a minimum of 130GB GPU memory to accommodate the entirety of its model weights within the GPU memory.

When using [model parallel](https://github.com/facebookresearch/fairscale/tree/main/fairscale/nn/model_parallel) on `MP = 1, 2, 4, 8` GPUs, you should divide the model to `MP` shards with `utils/convert_hf_weights_to_llama_ckpt.py`

```bash
python -u utils/convert_hf_weights_to_llama_ckpt.py \
    --base_model "/path/to/your/llama-65b-hf" \
    --lora_weights "/path/to/your/lora/weights" \
    --output_dir "/path/to/your/sharded_ckpt" \
    --total_ranks MP \
    --lora_target_modules='[q_proj,k_proj,v_proj,o_proj]' \
    --lora_r=16
```

When using model parallel on `MP = 3, 6, 12` GPUs, we recommend use `utils/convert_hf_weights_to_llama_expanded.py` to divide the checkpoint shards and install our customized `llama_dromedary` package for inference.

```bash
python -u utils/convert_hf_weights_to_llama_ckpt_expanded.py \
    --base_model "/path/to/your/llama-65b-hf" \
    --lora_weights "/path/to/your/lora/weights" \
    --output_dir "/path/to/your/sharded_ckpt" \
    --total_ranks MP \
    --target_att_dim 9216 \
    --target_ffn_dim 24576 \
    --target_vocab_size 36864 \
    --lora_target_modules='[q_proj,k_proj,v_proj,o_proj]' \
    --lora_r=16
```

For `MP = 5, 10` GPUs, here is the recommended expansion configuration for `llama_dromedary`.

```bash
python -u utils/convert_hf_weights_to_llama_ckpt_expanded.py \
    --base_model "/path/to/your/llama-65b-hf" \
    --lora_weights "/path/to/your/lora/weights" \
    --output_dir "/path/to/your/sharded_ckpt" \
    --total_ranks MP \
    --target_att_dim 8200 \
    --target_ffn_dim 22200 \
    --target_vocab_size 32000 \
    --lora_target_modules='[q_proj,k_proj,v_proj,o_proj]' \
    --lora_r=16
```
