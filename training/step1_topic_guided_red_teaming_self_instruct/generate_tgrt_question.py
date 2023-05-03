"""Self-instruction generation."""
import time
import json
import os
import sys
import random

import fire
import numpy as np
import tqdm
from pathlib import Path
from llama_dromedary.utils import load_model, setup_model_parallel, sync_model_parallel, llama_completion


def topic_guided_question_generation(
    generator,
    batch_prompts,
    meta_prompt,
    seed_questions,
    temperature,
    top_p,
    generate_max_len,
    num_examples=5,
):
  all_question_types = []
  all_topics = []
  all_questions = []
  seed_questions = seed_questions[:]

  for prompt in batch_prompts:
    random.shuffle(seed_questions)
    topics = prompt["topics"]
    question_types = prompt["question_types"]

    seed_topics = [data["topic"] for data in seed_questions[:num_examples]]
    seed_question_types = [data["question_type"] for data in seed_questions[:num_examples]]
    seed_example_questions = [data["question"] for data in seed_questions[:num_examples]]

    all_topics.append(seed_topics + topics)
    all_question_types.append(seed_question_types + question_types)
    all_questions.append(seed_example_questions)

  prompts = []
  expected_prefixs = []
  for question_types, topics, questions in zip(all_question_types, all_topics, all_questions):
    question_hints = [
      f"{i+1}. type: {question_type.replace('Questions', 'Instructions')}, topic: {topic}"
      for i, (question_type, topic) in enumerate(zip(question_types, topics))
    ]
    expected_prefixs.append(question_hints)
    question_hints = "\n" + "\n".join(question_hints)
    question_examples = [
      f"{i+1}. type: {question_type.replace('Questions', 'Instructions')}, topic: {topic}, instruction: {question}"
      for i, (question_type, topic, question) in enumerate(zip(question_types, topics, questions))
    ]
    question_examples = "\n" + "\n".join(question_examples)
    prompt = meta_prompt.format(question_hints, question_examples) + "\n"
    prompts.append(prompt)

  all_results = []
  eos_id = generator.tokenizer.eos_id
  results = llama_completion(
    generator,
    prompts=prompts,
    temperature=temperature,
    top_p=top_p,
    max_tokens=generate_max_len,
    logit_bias={eos_id: -100},  # prevent the </s> (eos token) token from being generated
  )

  for i, result in enumerate(results):
    result = result.strip()
    for sub_result in result.split("\n")[:-1]:
      sub_result = sub_result.strip()
      if sub_result == "":
        continue
      for prefix in expected_prefixs[i]:
        if sub_result.startswith(prefix):
          instruction = sub_result.split("instruction: ")[-1].strip()
          all_results.append(
            {
              "instruction": instruction,
              "topic": prefix.split("topic: ")[-1].strip(),
              "question_type": prefix.split("type: ")[-1].split(",")[0].strip(),
            }
          )
          break

  return all_results


def main(
    ckpt_dir: str,
    tokenizer_path: str,
    seed_questions_path: str,
    seed_topics_path: str,
    output_path: str,
    meta_prompt_file: str,
    num_instructions_to_generate: int = 15,
    request_batch_size: int = 32,
    num_examples: int = 5,
    temperature: float = 1.0,
    top_p: float = 1.0,
    max_seq_len: int = 512,
    max_shared_seq_len: int = 512,
    generate_max_len: int = 128,
    group_rank: int = -1,
    group_size: int = -1,
    seed: int = 42,
):
  assert group_rank >= 0 and group_size > 0, "Please specify group_rank and group_size"
  random.seed(seed + group_rank * 128)

  global_rank, world_size = setup_model_parallel()
  if global_rank > 0:
      sys.stdout = open(os.devnull, "w")

  meta_prompt = ""
  with open(meta_prompt_file) as f:
    meta_prompt = f.readlines()
    meta_prompt = "".join(meta_prompt)
    meta_prompt = meta_prompt.strip()

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
  sync_model_parallel()

  seed_questions = []
  with open(seed_questions_path) as f:
    for line in f:
      data = json.loads(line)
      seed_questions.append(data)

  seed_topics = []
  with open(seed_topics_path) as f:
    for line in f:
      data = json.loads(line)
      seed_topics.append(data)
  seed_topics = seed_topics[group_rank::group_size]
  original_seed_topics = seed_topics

  results = []
  if Path(output_path).exists():
      with open(output_path, "r") as f:
          results = f.readlines()
          results = [line for line in results if line.strip()]
      results = [json.loads(line) for line in results]

  seed_topics = [tuple(data.items()) for data in seed_topics]
  results = [tuple(
     {
        "topic": data["topic"],
        "question_type": data["question_type"],
     }.items()
  ) for data in results]
  results = set(results)
  seed_topics = [dict(_) for _ in seed_topics if _ not in results]

  print(
     "Original number of seed topics: %d, After removing the generated ones: %d" % (
          len(original_seed_topics), len(seed_topics))
  )

  output_handler = None
  if global_rank == 0:
      output_handler = open(output_path, "a")

  real_batch_size = request_batch_size * num_instructions_to_generate
  total_iters = len(seed_topics) // real_batch_size

  for i in tqdm.tqdm(range(0, len(seed_topics), real_batch_size)):
    batch_prompts = seed_topics[i:i+real_batch_size]
    new_batch_prompts = []
    for j in range(0, len(batch_prompts), num_instructions_to_generate):
      new_batch_prompts.append(
        {
          "topics": [data["topic"] for data in batch_prompts[j:j+num_instructions_to_generate]],
          "question_types": [data["question_type"] for data in batch_prompts[j:j+num_instructions_to_generate]],
        }
      )
    t0 = time.time()
    batch_prompts = new_batch_prompts
    results = topic_guided_question_generation(
      generator,
      batch_prompts,
      meta_prompt,
      seed_questions,
      temperature,
      top_p,
      generate_max_len,
      num_examples,
    )

    t1 = time.time()

    if group_rank == 0:
        print("=" * 20, "iter: ", i // real_batch_size, "/", total_iters, "latency: ", t1-t0)
        for output, idx in zip(results, range(8)):
            print(f"Output {idx} of {len(results)}: {output['question_type']}, {output['instruction']}")
            print()

    if output_handler is not None:
        for result in results:
            output_handler.write("\n" + json.dumps(result))
        output_handler.flush()

if __name__ == "__main__":
  fire.Fire(main)
