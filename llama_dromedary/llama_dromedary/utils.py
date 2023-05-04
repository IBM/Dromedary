"""Utilities for LLaMA Dromedary."""
from typing import Tuple, List, Dict, Optional
import os
import torch
import time
import json

from pathlib import Path

from fairscale.nn.model_parallel import initialize as mpu
from fairscale.nn.model_parallel.initialize import initialize_model_parallel

from .generation import LLaMA
from .model import ModelArgs, Transformer
from .tokenizer import Tokenizer


def setup_model_parallel() -> Tuple[int, int]:
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    global_rank = int(os.environ.get("RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", -1))

    torch.distributed.init_process_group("nccl")
    initialize_model_parallel(world_size, pipeline_length=1)
    print("Model parallelism:", mpu.get_model_parallel_world_size())
    print("Global rank:", global_rank, "World size:", world_size)
    torch.cuda.set_device(local_rank)

    # seed must be the same in all processes
    torch.manual_seed(1)
    return global_rank, world_size


def sync_model_parallel():
    torch.distributed.barrier()


def load_model(
    ckpt_dir: str,
    tokenizer_path: str,
    global_rank: int,
    world_size: int,
    max_seq_len: int,
    max_batch_size: int,
    max_shared_seq_len: int,
    disable_cache: bool = False,
) -> LLaMA:
    start_time = time.time()
    checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
    assert world_size == len(
        checkpoints
    ), f"Loading a checkpoint for MP={len(checkpoints)} but world size is {world_size}"
    ckpt_path = checkpoints[global_rank]
    print("Loading")
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    with open(Path(ckpt_dir) / "params.json", "r") as f:
        params = json.loads(f.read())

    model_args: ModelArgs = ModelArgs(
        max_seq_len=max_seq_len, max_batch_size=max_batch_size, **params
    )
    tokenizer = Tokenizer(model_path=tokenizer_path)
    model_args.vocab_size = tokenizer.n_words

    if model_args.qkv_dim != 0:
        print("Original n_heads:", model_args.n_heads)
        model_args.n_heads = (model_args.n_heads * model_args.qkv_dim) // model_args.dim
        print("New n_heads:", model_args.n_heads)

    model_args.max_shared_seq_len = max_shared_seq_len
    model_args.use_prefix_cache = True
    model_args.disable_cache = disable_cache

    torch.set_default_tensor_type(torch.cuda.HalfTensor)
    model = Transformer(model_args)
    model.load_state_dict(checkpoint, strict=False)
    model.eval()
    model.half()

    generator = LLaMA(model, tokenizer)
    print(f"Loaded in {time.time() - start_time:.2f} seconds")
    return generator


def llama_completion(
    generator: LLaMA,
    prompts: List[str],
    temperature: float = 0.0,
    top_p: float = 1.0,
    max_tokens: int = 1024,
    logit_bias: Optional[Dict[int, float]] = None,
    stop: Optional[str] = None,
    unitoken_frequency_penalty: float = 0.0,
    bitoken_frequency_penalty: float = 0.0,
    tritoken_frequency_penalty: float = 0.0,
    quadtoken_frequency_penalty: float = 0.0,
) -> list:
    """
    Generate completions for a batch of prompts.

    Args:
        generator: LLaMA generator
        prompts: batch of prompts
        temperature: temperature for sampling
        top_p: top_p for sampling
        max_tokens: maximum number of tokens to generate
        logit_bias: dictionary of token ids to bias logits
        stop: stop string
        unitoken_frequency_penalty: frequency penalty for unigrams
        bitoken_frequency_penalty: frequency penalty for bigrams
        tritoken_frequency_penalty: frequency penalty for trigrams
        quadtoken_frequency_penalty: frequency penalty for quadgrams

    Returns:
        list of completions
    """
    outputs = generator.generate(
        prompts,
        max_gen_len=max_tokens,
        temperature=temperature,
        top_p=top_p,
        logit_bias=logit_bias,
        stop=stop,
        unitoken_frequency_penalty=unitoken_frequency_penalty,
        bitoken_frequency_penalty=bitoken_frequency_penalty,
        tritoken_frequency_penalty=tritoken_frequency_penalty,
        quadtoken_frequency_penalty=quadtoken_frequency_penalty,
    )

    results = []
    for output in outputs:
        if stop is not None:
            for stop_string in stop:
                if stop_string in output:
                    output = output[: output.index(stop_string)]
        results.append(output)

    return results


def llama_scoring(
    generator: LLaMA,
    prompts: List[str],
    targets: List[str],
    temperature: float = 1.0,
    logit_bias: Optional[Dict[int, float]] = None,
) -> List[float]:
    """
    Compute scores for a batch of prompts and targets.

    Args:
        generator: LLaMA generator
        prompts: batch of prompts
        targets: batch of targets
        temperature: temperature for scaling logits
        logit_bias: dictionary of token ids to bias logits

    Returns:
        list of scores
    """
    scores = generator.score(
        prompts,
        targets,
        temperature=temperature,
        logit_bias=logit_bias,
    )

    return scores
