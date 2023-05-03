"""Topic generation for 20 question types."""
import time
import json
import os
import sys
import random
import string

import tqdm
import fire
from llama_dromedary.utils import load_model, setup_model_parallel, sync_model_parallel, llama_completion


def brainstorm_topics(
    generator,
    batch_prompts,
    meta_prompt,
    temperature,
    top_p,
    generate_max_len,
):
  question_types = []
  all_topics = []

  for prompt in batch_prompts:
    topics = prompt["topics"]
    question_type = prompt["question_type"]
    question_types.append(question_type.strip())
    all_topics.append(topics)

  all_results = []

  prompts = []
  for question_type, topics in zip(question_types, all_topics):
    prompt = meta_prompt.format(question_type) + "\n\n"
    for i, topic in enumerate(topics):
      prompt += f"{i+1}. {topic}\n"
    prompts.append(prompt)

  eos_id = generator.tokenizer.eos_id
  results = llama_completion(
    generator,
    prompts=prompts,
    temperature=temperature,
    top_p=top_p,
    max_tokens=generate_max_len,
    logit_bias={eos_id: -100},  # prevent the </s> (eos token) token from being generated
  )

  for question_type, result in zip(question_types, results):
    single_results = result.split("\n")[:20]
    formatted_single_results = []
    for single_result in single_results:
      try:
        single_result = single_result.split('. ', 1)[-1]
        if len(single_result.split()) > 3:
          raise Exception("Too long")
        if len(single_result.strip()) == 0:
          raise Exception("Too short")
        # the result should start with a Capital letter
        if single_result[0] not in string.ascii_uppercase:
          raise Exception("Not capitalized")
        # the result should not end with a punctuation
        if single_result[-1] in string.punctuation:
          raise Exception("Ends with punctuation")
        single_result = {
          "topic": single_result,
          "question_type": question_type,
        }
        formatted_single_results.append(single_result)
      except:
        pass
    all_results.extend(formatted_single_results)

  return all_results


def deduplicate_strings(strings):
  """
  Deduplicates a list of strings based on their lowercase versions.

  Args:
      strings: A list of strings.

  Returns:
      A deduplicated list of strings.
  """
  seen = set()
  deduplicated = []

  for string in strings:
    lowered = string.lower()
    if lowered not in seen:
      seen.add(lowered)
      deduplicated.append(string)
  return deduplicated


def main(
    ckpt_dir: str,
    tokenizer_path: str,
    seed_questions_path: str,
    output_path: str,
    meta_prompt_file: str,
    request_batch_size: int = 32,
    num_examples: int = 5,
    temperature: float = 1.0,
    top_p: float = 1.0,
    starting_round: int = 0,
    generation_epoch: int = 1,
    max_seq_len: int = 512,
    max_shared_seq_len: int = 512,
    generate_max_len: int = 128,
    seed: int = 42,
):
  random.seed(seed)

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
  print("Model loading time: ", loading_time)
  sync_model_parallel()

  for round in range(generation_epoch):
    print(f"Generation Epoch {round}")
    if round < starting_round:
      continue

    type2topic = {}

    if round == 0:
      seed_question_file = seed_questions_path
    else:
      seed_question_file = output_path.replace(".jsonl", f"_epoch{round-1}.jsonl")
    print("Seed question file:", seed_question_file)

    with open(seed_question_file) as f:
      prompts = f.readlines()
      prompts = [json.loads(prompt.strip()) for prompt in prompts if prompt.strip()]
      for line in prompts:
        if line:
          question_type = line["question_type"]
          topic = line["topic"]
          if question_type not in type2topic:
            type2topic[question_type] = []
          type2topic[question_type].append(topic)

    min_num_deduplicated_topics = 1048576
    for question_type in type2topic:
      print("Question Type:", question_type)
      num_topics = len(type2topic[question_type])

      deduplicated_topics = deduplicate_strings(type2topic[question_type])
      num_deduplicated_topics = len(deduplicated_topics)
      min_num_deduplicated_topics = min(min_num_deduplicated_topics, num_deduplicated_topics)

      print(f"Number of topics: {num_topics}")
      print(f"Number of deduplicated topics: {num_deduplicated_topics}")
      type2topic[question_type] = deduplicated_topics

    # normalize the distribution of topics
    for question_type in type2topic:
      type2topic[question_type] = type2topic[question_type][:min_num_deduplicated_topics]

    print("Total types:", len(type2topic))
    print("\n\n" + "=" * 20 + "\n\n")
    all_topics_and_types = []
    for question_type in type2topic:
        topics = type2topic[question_type]
        if len(topics) < num_examples * 2:
          for i in range(0, len(topics)):
            all_topics_and_types.append({"topics": [topics[i]], "question_type": question_type})
        else:
          for i in range(0, len(topics)):
            random.shuffle(topics)
            all_topics_and_types.append({"topics": topics[:num_examples], "question_type": question_type})

    with open(output_path.replace(".jsonl", f"_epoch{round}.jsonl"), "w") as f:
      for question_type in type2topic:
        for topic in type2topic[question_type]:
          f.write(json.dumps({"topic": topic, "question_type": question_type}) + "\n")

      for i in tqdm.tqdm(range(0, len(all_topics_and_types), request_batch_size)):
        batch_prompts = all_topics_and_types[i:i + request_batch_size]
        new_topics = brainstorm_topics(
          generator,
          batch_prompts,
          meta_prompt,
          temperature,
          top_p,
          generate_max_len,
        )

        for new_topic in new_topics:
          f.write(json.dumps(new_topic) + "\n")
        
        f.flush()


if __name__ == "__main__":
  fire.Fire(main)
