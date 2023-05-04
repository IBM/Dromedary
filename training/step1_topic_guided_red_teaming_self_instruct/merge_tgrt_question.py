"""
Merge the questions in the TGRT dataset.

run:
python merge_tgrt_question.py \
  --data_file_pattern "/path/to/your/llama65b_tgrt_questions_8shards_*.jsonl" \
  --output_file /path/to/your/llama65b_tgrt_questions_merged.json
"""

import random
import json
import glob

import fire


def main(
  data_file_pattern: str,
  output_file: str,
):
  all_topics = set()
  num_original_topics = 0
  results = []

  for data_file in glob.glob(data_file_pattern):
    with open(data_file) as f:
      prompts = f.readlines()
      prompts = [json.loads(prompt.strip()) for prompt in prompts if prompt.strip()]
      num_original_topics += len(prompts)
      for line in prompts:
        if (line["topic"], line["question_type"]) not in all_topics:
          all_topics.add((line["topic"], line["question_type"]))
          new_line = {}
          new_line["instruction"] = line["instruction"]
          new_line["input"] = ""
          new_line["output"] = ""
          results.append(new_line)

  print("Number of original topics:", num_original_topics)
  print("Number of results:", len(results))

  random.shuffle(results)
  with open(output_file, "w") as f:
    f.write(json.dumps(results))


if __name__ == "__main__":
  fire.Fire(main)
