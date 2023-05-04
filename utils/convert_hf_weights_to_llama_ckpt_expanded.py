import json
import os

import fire
import numpy as np
import torch
import transformers
from peft import PeftModel
from typing import List, Optional


assert (
    "LlamaTokenizer" in transformers._import_structure["models.llama"]
), "LLaMA is now in HuggingFace's main branch.\nPlease reinstall it: pip uninstall transformers && pip install git+https://github.com/huggingface/transformers.git"  # noqa: E501

from transformers import LlamaForCausalLM, AutoConfig

from peft import (  # noqa: E402
    LoraConfig,
    get_peft_model,
    set_peft_model_state_dict,
)


def translate_state_dict_key(k):  # noqa: C901
    k = k.replace("base_model.model.", "")
    if k == "model.embed_tokens.weight":
        return "tok_embeddings.weight"
    elif k == "model.norm.weight":
        return "norm.weight"
    elif k == "lm_head.weight":
        return "output.weight"
    elif k.startswith("model.layers."):
        layer = k.split(".")[2]
        if k.endswith(".self_attn.q_proj.weight"):
            return f"layers.{layer}.attention.wq.weight"
        elif k.endswith(".self_attn.k_proj.weight"):
            return f"layers.{layer}.attention.wk.weight"
        elif k.endswith(".self_attn.v_proj.weight"):
            return f"layers.{layer}.attention.wv.weight"
        elif k.endswith(".self_attn.o_proj.weight"):
            return f"layers.{layer}.attention.wo.weight"
        elif k.endswith(".mlp.gate_proj.weight"):
            return f"layers.{layer}.feed_forward.w1.weight"
        elif k.endswith(".mlp.down_proj.weight"):
            return f"layers.{layer}.feed_forward.w2.weight"
        elif k.endswith(".mlp.up_proj.weight"):
            return f"layers.{layer}.feed_forward.w3.weight"
        elif k.endswith(".input_layernorm.weight"):
            return f"layers.{layer}.attention_norm.weight"
        elif k.endswith(".post_attention_layernorm.weight"):
            return f"layers.{layer}.ffn_norm.weight"
        elif k.endswith("rotary_emb.inv_freq") or "lora" in k:
            return None
        else:
            print(layer, k)
            raise NotImplementedError
    else:
        print(k)
        raise NotImplementedError


def shard_weights(k, v, rank, total_ranks):
    def shard_dim(total_size):
        # shard size should be divisible by 64
        multiple_of = 8
        shard_size = total_size // total_ranks
        shard_size = multiple_of * ((shard_size + multiple_of - 1) // multiple_of)
        return shard_size

    if "wo" in k or "w2" in k:
        # split in the second demension
        total_dims = v.shape[1]
        shard_size = shard_dim(total_dims)
        start = rank * shard_size
        end = min((rank + 1) * shard_size, total_dims)
        return v[:, start:end].clone()

    elif "tok_embeddings" in k or "output" in k or "wq" in k or "wk" in k or "wv" in k or "w1" in k or "w3" in k:
        # split in the first demension
        total_dims = v.shape[0]
        shard_size = shard_dim(total_dims)
        start = rank * shard_size
        end = min((rank + 1) * shard_size, total_dims)
        return v[start:end, :].clone()

    elif "norm" in k or "rope" in k:
        # do not shard
        return v

    else:
        raise NotImplementedError


def expand_weights(k, v, expanded_att_dim, expanded_ffn_dim, expanded_vocab_size):
    if "wq" in k or "wk" in k or "wv" in k:
        v_dim_0 = v.shape[0]
        v_dim_1 = v.shape[1]

        assert expanded_att_dim >= v_dim_0
        if expanded_att_dim == v_dim_0:
            return v
        new_v = torch.zeros(
            expanded_att_dim - v_dim_0, v_dim_1, dtype=v.dtype, device=v.device
        )
        new_v = torch.concat([v, new_v], dim=0)
        return new_v

    elif "wo" in k:
        v_dim_0 = v.shape[0]
        v_dim_1 = v.shape[1]

        assert expanded_att_dim >= v_dim_1
        if expanded_att_dim == v_dim_1:
            return v
        new_v = torch.zeros(
            v_dim_0, expanded_att_dim - v_dim_1, dtype=v.dtype, device=v.device
        )
        new_v = torch.concat([v, new_v], dim=1)
        return new_v

    elif "w1" in k or "w3" in k:
        v_dim_0 = v.shape[0]
        v_dim_1 = v.shape[1]

        assert expanded_ffn_dim >= v_dim_1
        if expanded_ffn_dim == v_dim_1:
            return v
        new_v = torch.zeros(
            expanded_ffn_dim - v_dim_0, v_dim_1, dtype=v.dtype, device=v.device
        )
        new_v = torch.concat([v, new_v], dim=0)
        return new_v

    elif "w2" in k:
        v_dim_0 = v.shape[0]
        v_dim_1 = v.shape[1]

        assert expanded_ffn_dim >= v_dim_0
        if expanded_ffn_dim == v_dim_1:
            return v
        new_v = torch.zeros(
            v_dim_0, expanded_ffn_dim - v_dim_1, dtype=v.dtype, device=v.device
        )
        new_v = torch.concat([v, new_v], dim=1)
        return new_v

    elif "tok_embeddings" in k or "output" in k:
        v_dim_0 = v.shape[0]
        v_dim_1 = v.shape[1]

        assert expanded_vocab_size >= v_dim_0
        if expanded_vocab_size == v_dim_0:
            return v
        new_v = torch.zeros(
            expanded_vocab_size - v_dim_0, v_dim_1, dtype=v.dtype, device=v.device
        )
        new_v = torch.concat([v, new_v], dim=0)
        return new_v

    else:
        return v



def main(
    base_model: str = "",
    lora_weights: str = "none",
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_target_modules: List[str] = [
        "q_proj",
        "v_proj",
    ],
    output_dir: str = None,
    total_ranks: int = 1,
    write_mode: bool = True,
    expanded_att_dim: int = 0,
    expanded_ffn_dim: int = 0,
    expanded_vocab_size: int = 0,
):
    if output_dir is None:
        raise ValueError("output_dir must be specified")

    if expanded_att_dim == 0 or expanded_ffn_dim == 0 or expanded_vocab_size == 0:
        raise ValueError("expanded_att_dim, expanded_ffn_dim, expanded_vocab_size must be specified")


    if lora_weights == "none":
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.float16,
            device_map={"": "cpu"},
        )
        lora_model = model
    else:
        checkpoint_name = os.path.join(
            lora_weights, "adapter_model.bin"
        )
        print(checkpoint_name)
        adapters_weights = torch.load(checkpoint_name)

        model = LlamaForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.float16,
            device_map={"": "cpu"},
        )

        config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=lora_target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        lora_model = get_peft_model(model, config)
        tmp_lora_model = set_peft_model_state_dict(lora_model, adapters_weights)
        if tmp_lora_model is not None:
            lora_model = tmp_lora_model

        # merge weights
        for layer in lora_model.base_model.model.model.layers:
            if "q_proj" in lora_target_modules:
                layer.self_attn.q_proj.merge_weights = True
            if "v_proj" in lora_target_modules:
                layer.self_attn.v_proj.merge_weights = True
            if "k_proj" in lora_target_modules:
                layer.self_attn.k_proj.merge_weights = True
            if "o_proj" in lora_target_modules:
                layer.self_attn.o_proj.merge_weights = True
            if "gate_proj" in lora_target_modules:
                layer.mlp.gate_proj.merge_weights = True
            if "down_proj" in lora_target_modules:
                layer.mlp.down_proj.merge_weights = True
            if "up_proj" in lora_target_modules:
                layer.mlp.up_proj.merge_weights = True

    lora_model.train(False)

    lora_model_sd = lora_model.state_dict()

    model_config = AutoConfig.from_pretrained(base_model)

    params = {
        "dim": model_config.hidden_size,
        "multiple_of": 256,
        "n_heads": model_config.num_attention_heads,
        "n_layers": model_config.num_hidden_layers,
        "norm_eps": model_config.rms_norm_eps,
        "vocab_size": -1,
        "qkv_dim": expanded_att_dim,
        "ffn_dim": expanded_ffn_dim,
        "model_vocab_size": expanded_vocab_size,
    }
    n_heads = params["n_heads"]
    dim = params["dim"]


    def unpermute(w):
        return (
            w.view(n_heads, 2, dim // n_heads // 2, dim)
            .transpose(1, 2)
            .reshape(dim, dim)
        )

    os.makedirs(output_dir, exist_ok=False)
    print("Making output directory: ", output_dir)
    with open(os.path.join(output_dir, "params.json"), "w") as f:
        json.dump(params, f)

    for rank in range(total_ranks):
        new_state_dict = {}

        model_params_count = 0

        for k, v in lora_model_sd.items():
            new_k = translate_state_dict_key(k)
            if new_k is not None:
                if "wq" in new_k or "wk" in new_k:
                    new_v = unpermute(v)
                else:
                    new_v = v

                new_v = expand_weights(new_k, new_v,
                                       expanded_att_dim,
                                       expanded_ffn_dim,
                                       expanded_vocab_size)

                new_v = shard_weights(new_k, new_v, rank, total_ranks)
                if "layers" not in new_k or "layers.0" in new_k:
                    print(f"{new_k},", "shape:", new_v.shape, "dtype:", new_v.dtype)

                v_np = new_v.cpu().numpy()
                new_v = torch.from_numpy(v_np.astype(np.float16))

                new_state_dict[new_k] = new_v
                model_params_count += new_v.numel()

        print(f"Total model params: {model_params_count}")
        print(f"Estimated storage: {model_params_count * 2 / 1024 / 1024 / 1024:.2f} GB")
        if write_mode:
            print(f"Saving to: {os.path.join(output_dir, f'consolidated.{rank:02d}.pth')}")
            torch.save(new_state_dict, os.path.join(output_dir, f"consolidated.{rank:02d}.pth"))
        else:
            print(f"Debug: saving to: {os.path.join(output_dir, f'consolidated.{rank:02d}.pth')}")


if __name__ == "__main__":
    fire.Fire(main)
