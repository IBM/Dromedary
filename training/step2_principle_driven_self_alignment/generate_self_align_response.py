"""Self-aligned response generation."""
import math
import os
import sys
import fire
import time
import tqdm
import json

from pathlib import Path
from llama_dromedary.utils import setup_model_parallel, sync_model_parallel, load_model, llama_completion


def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 1.0,
    top_p: float = 1.0,
    max_seq_len: int = 512,
    max_batch_size: int = 32,
    max_shared_seq_len: int = 512,
    generate_max_len: int = 128,
    group_rank: int = -1,
    group_size: int = -1,
    input_file: str = None,
    output_file: str = None,
    meta_prompt_file: str = None,
    unitoken_frequency_penalty: float = 0.0,
    bitoken_frequency_penalty: float = 0.0,
    tritoken_frequency_penalty: float = 0.0,
    quadtoken_frequency_penalty: float = 0.0,
):
    assert group_rank >= 0, "Must specify group rank"
    assert group_size >= 0, "Must specify group size"
    assert input_file is not None and output_file is not None, "Must specify input and output files"
    assert meta_prompt_file is not None, "Must specify meta prompt file"

    with open(input_file, "r") as f:
        inputs = json.load(f)
    inputs = inputs[group_rank::group_size]

    generate_prompt_fn = generate_prompt
    with open(meta_prompt_file, "r") as f:
        data = f.readlines()
    meta_prompt = "".join(data)
    meta_prompt = meta_prompt.strip()

    global_rank, world_size = setup_model_parallel()
    if global_rank > 0:
        sys.stdout = open(os.devnull, "w")

    t0 = time.time()
    generator = load_model(
        ckpt_dir, tokenizer_path, global_rank, world_size, max_seq_len, max_batch_size, max_shared_seq_len,
    )
    t1 = time.time()
    loading_time = t1-t0
    print("Model loading time on %d: " % group_size, loading_time)

    results = []
    # record current progress
    if Path(output_file).exists():
        with open(output_file, "r") as f:
            results = f.readlines()
            results = [line for line in results if len(line.strip()) > 0]

    inputs = inputs[len(results):]
    print("Skipping %d examples" % len(results))

    batching_inputs = tqdm.tqdm(BatchIterator(inputs, max_batch_size), desc="Batched inference", disable=global_rank > 0)
    total_iters = len(inputs) // max_batch_size

    output_handler = None
    if global_rank == 0:
        output_handler = open(output_file, "a")

    sync_model_parallel()

    logit_bias = {
        generator.tokenizer.encode("Sorry", bos=False, eos=False)[0]: -100,
    }
    print("logit_bias: ", logit_bias)

    # prepare inputs with batch size $max_batch_size
    for iter, batched_inputs in enumerate(batching_inputs):
        t0 = time.time()
        prompts = [generate_prompt_fn(meta_prompt, ex_input["instruction"], ex_input["input"])
                          for ex_input in batched_inputs]

        outputs = llama_completion(
            generator,
            prompts,
            max_tokens=generate_max_len,
            temperature=temperature,
            top_p=top_p,
            logit_bias=logit_bias,
            unitoken_frequency_penalty=unitoken_frequency_penalty,
            bitoken_frequency_penalty=bitoken_frequency_penalty,
            tritoken_frequency_penalty=tritoken_frequency_penalty,
            quadtoken_frequency_penalty=quadtoken_frequency_penalty,
        )

        t1 = time.time()

        results = []
        for ex_input, output in zip(batched_inputs, outputs):
            results.append({"instruction": ex_input["instruction"], "input": ex_input["input"], "output": output})

        if group_rank == 0:
            for ex_input, output, _ in zip(batched_inputs, outputs, range(8)):
                print("=" * 20, "iter: ", iter, "/", total_iters, "latency: ", t1-t0)
                print(f"Input: {ex_input['instruction']}: {ex_input['input']}")
                print(f"Output: {output}")
                print()

        if output_handler is not None:
            for result in results:
                output_handler.write("\n" + json.dumps(result))
            output_handler.flush()


class BatchIterator:
    def __init__(self, data, batch_size=1):
        self.data = data
        self.batch_size = batch_size
        self.index = 0

    def __iter__(self):
        for i in range(0, len(self.data), self.batch_size):
            yield self.data[i:i + self.batch_size]

    def __len__(self):
        return math.ceil(len(self.data) / self.batch_size)


def generate_prompt(meta_prompt, instruction, input=None):
    if input:
        return f"""{meta_prompt} {instruction}

{input}

Watson (internal thoughts):"""
    else:
        return f"""{meta_prompt} {instruction}

Watson (internal thoughts):"""


if __name__ == "__main__":
    fire.Fire(main)
