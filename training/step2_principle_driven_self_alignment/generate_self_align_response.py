"""Self-aligned response generation."""
import math
import os
import fire
import time
import tqdm
import json

from pathlib import Path
from llama_dromedary import Llama


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
):
    assert group_rank >= 0, "Must specify group rank"
    assert group_size >= 0, "Must specify group size"
    assert (
        input_file is not None and output_file is not None
    ), "Must specify input and output files"
    assert meta_prompt_file is not None, "Must specify meta prompt file"

    with open(input_file, "r") as f:
        inputs = json.load(f)
    inputs = inputs[group_rank::group_size]

    generate_prompt_fn = generate_prompt
    with open(meta_prompt_file, "r") as f:
        meta_prompt = f.read().strip()

    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
        max_shared_seq_len=max_shared_seq_len,
    )

    results = []
    # record current progress
    if "shards" not in output_file and group_size > 1:
        output_file = output_file.replace(
            ".json", f"_{group_size}shards_{group_rank}.json"
        )

    if Path(output_file).exists():
        with open(output_file, "r") as f:
            results = f.readlines()
            results = [line for line in results if len(line.strip()) > 0]

    inputs = inputs[len(results) :]
    print("Skipping %d examples" % len(results))

    global_rank = int(os.environ.get("RANK", "0"))
    batching_inputs = tqdm.tqdm(
        BatchIterator(inputs, max_batch_size),
        desc="Batched inference",
        disable=global_rank > 0,
    )
    total_iters = len(inputs) // max_batch_size

    output_handler = None
    if global_rank == 0:
        output_handler = open(output_file, "a")

    # prepare inputs with batch size $max_batch_size
    for iter, batched_inputs in enumerate(batching_inputs):
        t0 = time.time()
        prompts = [
            generate_prompt_fn(
                meta_prompt, ex_input["instruction"].strip(), ex_input["input"].strip()
            )
            for ex_input in batched_inputs
        ]

        outputs = generator.text_completion(
            prompts,
            max_gen_len=generate_max_len,
            temperature=temperature,
            top_p=top_p,
        )

        t1 = time.time()

        results = []
        for ex_input, output in zip(batched_inputs, outputs):
            results.append(
                {
                    "instruction": ex_input["instruction"],
                    "input": ex_input["input"],
                    "output": output,
                }
            )

        if group_rank == 0:
            for ex_input, output, _ in zip(batched_inputs, outputs, range(8)):
                print("=" * 20, "iter: ", iter, "/", total_iters, "latency: ", t1 - t0)
                print(f"Input: {ex_input['instruction']}: {ex_input['input']}")
                print(f"Output: {output}")
                print()

        if output_handler is not None:
            for result in results:
                output_handler.write(json.dumps(result) + "\n")
            output_handler.flush()


class BatchIterator:
    def __init__(self, data, batch_size=1):
        self.data = data
        self.batch_size = batch_size
        self.index = 0

    def __iter__(self):
        for i in range(0, len(self.data), self.batch_size):
            yield self.data[i : i + self.batch_size]

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
