"""Self-instruction generation. Adapted from [https://github.com/tatsu-lab/stanford_alpaca/blob/main/generate_instruction.py]"""
import time
import json
import os
import sys
import random
import re
import string
from functools import partial
from multiprocessing import Pool

import fire
import tqdm
import numpy as np
from rouge_score import rouge_scorer
from llama_dromedary.utils import load_model, setup_model_parallel, sync_model_parallel, llama_completion


def encode_prompt(prompt_instructions, meta_prompt):
    """Encode multiple prompt instructions into a single string."""
    prompt = meta_prompt + "\n"

    for idx, task_dict in enumerate(prompt_instructions):
        instruction, input = task_dict["instruction"], task_dict["input"]
        instruction = re.sub(r"\s+", " ", instruction).strip().rstrip(":")
        input = "<noinput>" if input.lower() == "" else input
        prompt += f"###\n"
        prompt += f"{idx + 1}. Instruction: {instruction}\n"
        prompt += f"{idx + 1}. Input:\n{input}\n"
    prompt += f"###\n"
    prompt += f"{idx + 2}. Instruction:"
    return prompt


def post_process_gpt3_response(num_prompt_instructions, response):
    raw_instructions = f"{num_prompt_instructions+1}. Instruction:" + response
    raw_instructions = re.split("###", raw_instructions)
    # if the decoding stops due to length, the last example is likely truncated so we discard it
    raw_instructions = raw_instructions[:-1]
    instructions = []
    for idx, inst in enumerate(raw_instructions):
        idx += num_prompt_instructions + 1
        splitted_data = re.split(f"{idx}\.\s+(Instruction|Input):", inst)
        if len(splitted_data) != 5:
            continue
        else:
            inst = splitted_data[2].strip()
            input = splitted_data[4].strip()
            input = "" if input.lower() == "<noinput>" else input
            output = ""
        # filter out too short or too long instructions
        if len(inst.split()) <= 3 or len(inst.split()) > 150:
            continue
        # filter based on keywords that are not suitable for language models.
        blacklist = [
            "image",
            "images",
            "graph",
            "graphs",
            "picture",
            "pictures",
            "file",
            "files",
            "map",
            "maps",
            "draw",
            "plot",
            "go to",
            "video",
            "audio",
            "music",
            "flowchart",
            "diagram",
        ]
        if any(find_word_in_string(word, inst) for word in blacklist):
            continue
        if any(find_word_in_string(word, input) for word in blacklist):
            continue
        # filter those starting with punctuation
        if inst[0] in string.punctuation:
            continue
        # filter those starting with non-english character
        if not inst[0].isascii():
            continue
        instructions.append({"instruction": inst, "input": input, "output": output})
    return instructions


def find_word_in_string(w, s):
    return re.compile(r"\b({0})\b".format(w), flags=re.IGNORECASE).search(s)


def main(
    ckpt_dir: str,
    tokenizer_path: str,
    seed_tasks_path: str,
    output_path: str,
    meta_prompt_file: str,
    num_instructions_to_generate=100,
    num_prompt_instructions=3,
    request_batch_size=32,
    temperature=1.0,
    top_p=1.0,
    num_cpus=16,
    seed=42,
    max_seq_len: int = 512,
    max_shared_seq_len: int = 512,
    generate_max_len: int = 128,
    group_rank: int = -1,
    group_size: int = -1,
):
    assert group_rank >= 0 and group_size > 0, "Please specify group_rank and group_size"

    global_rank, world_size = setup_model_parallel()
    if global_rank > 0:
        sys.stdout = open(os.devnull, "w")

    with open(seed_tasks_path, "r") as f:
        seed_tasks = [json.loads(l) for l in f]
    seed_instruction_data = [
        {
            "instruction": t["instruction"],
            "input": t["instances"][0]["input"],
            "output": t["instances"][0]["output"],
        } for t in seed_tasks
    ]
    print(f"Loaded {len(seed_instruction_data)} human-written seed instructions")

    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    resume_epoch = 0
    resume_step = 0
    machine_instruction_data = []
    last_epoch_instructions = []
    on_epoch_instructions = []

    with open(meta_prompt_file, "r") as f:
        meta_prompt = f.read()

    target_file = output_path.replace(".json", f"_epoch{epoch}.json")
    results = []
    # record current progress
    if os.path.exists(target_file):
        with open(target_file, "r") as f:
            results = f.readlines()
            results = [json.loads(line) for line in results if len(line.strip()) > 0]
            print(f"Loaded {len(results)} machine-written instructions from {target_file}")
            resume_epoch = epoch
            resume_step = len(results)
            machine_instruction_data.extend(results)
            last_epoch_instructions = on_epoch_instructions
            on_epoch_instructions = results

    t0 = time.time()
    generator = load_model(
        ckpt_dir,
        tokenizer_path,
        global_rank,
        world_size,
        max_seq_len,
        request_batch_size,
        max_shared_seq_len,
    )
    t1 = time.time()
    loading_time = t1-t0
    print("Model loading time on %d: " % group_size, loading_time)

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    eos_id = generator.tokenizer.eos_id
    sync_model_parallel()

    epoch = resume_epoch
    # now let's generate new instructions!
    random.seed(seed + group_rank * 128 + epoch * 16384)

    progress_bar = tqdm.tqdm(total=num_instructions_to_generate, desc=f"Epoch {epoch}", disable=global_rank > 0)
    if resume_step:
        progress_bar.update(resume_step)

    output_handler = None
    target_file = output_path.replace(".json", f"_epoch{epoch}.json")
    if global_rank == 0:
        output_handler = open(target_file, "a")

    # first we tokenize all the seed instructions and generated machine instructions
    all_instructions = [d["instruction"] for d in seed_instruction_data] + [
        d["instruction"] for d in machine_instruction_data
    ]
    all_instruction_tokens = [scorer._tokenizer.tokenize(inst) for inst in all_instructions]

    on_epoch_seed_instruction_data = seed_instruction_data + last_epoch_instructions
    last_epoch_instructions = []
    print(f"Using {len(on_epoch_seed_instruction_data)} seed instructions for epoch {epoch}")

    request_idx = len(on_epoch_instructions)
    while len(on_epoch_instructions) < num_instructions_to_generate:
        request_idx += 1

        pre_process_start = time.time()
        batch_inputs = []
        for _ in range(request_batch_size):
            # only sampling from the seed tasks
            prompt_instructions = random.sample(on_epoch_seed_instruction_data, num_prompt_instructions)
            prompt = encode_prompt(prompt_instructions, meta_prompt=meta_prompt)
            batch_inputs.append(prompt)
        pre_process_duration = time.time() - pre_process_start

        request_start = time.time()
        results = llama_completion(
            generator,
            prompts=batch_inputs,
            temperature=temperature,
            top_p=top_p,
            max_tokens=generate_max_len,
            logit_bias={eos_id: -100},  # prevent the </s> (eos token) token from being generated
        )
        request_duration = time.time() - request_start

        process_start = time.time()

        instruction_data = []
        for result in results:
            new_instructions = post_process_gpt3_response(num_prompt_instructions, result)
            instruction_data += new_instructions

        total = len(instruction_data)
        keep = 0
        results = []
        with Pool(num_cpus) as p:
            for instruction_data_entry in instruction_data:
                # computing similarity with the pre-tokenzied instructions
                new_instruction_tokens = scorer._tokenizer.tokenize(instruction_data_entry["instruction"])
                rouge_scores = p.map(
                    partial(rouge_scorer._score_lcs, new_instruction_tokens),
                    all_instruction_tokens,
                )
                rouge_scores = [score.fmeasure for score in rouge_scores]
                if max(rouge_scores) > 0.7:
                    continue
                else:
                    keep += 1
                instruction_data_entry["avg_similarity_score"] = float(np.mean(rouge_scores))
                machine_instruction_data.append(instruction_data_entry)
                on_epoch_instructions.append(instruction_data_entry)
                results.append(instruction_data_entry)
                all_instructions.append(instruction_data_entry["instruction"])
                all_instruction_tokens.append(new_instruction_tokens)
                progress_bar.update(1)
        process_duration = time.time() - process_start

        if group_rank == 0:
            for instruction_data_entry in instruction_data[:5]:
                print("=" * 10, "iter:", len(on_epoch_instructions), "/", num_instructions_to_generate, "=" * 10)
                print(f"Instruction: {instruction_data_entry['instruction']}")
                print(f"Input: {instruction_data_entry['input']}")
            print()
            print(f"Request {request_idx} took {request_duration:.2f}s, processing took {pre_process_duration:.2f}s + {process_duration:.2f}s")
            print(f"Generated {total} instructions, kept {keep} instructions")
            print()

        if output_handler is not None:
            for result in results:
                output_handler.write("\n" + json.dumps(result))
            output_handler.flush()


if __name__ == "__main__":
    fire.Fire(main)
