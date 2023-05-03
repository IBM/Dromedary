"""
Merge the data from the vanilla self-instruct dataset.

run:
python merge_self_instruct.py \
  --data_file_pattern "/path/to/your/llama65b_self_instruct_32shards_*.jsonl" \
  --output_file /path/to/your/llama65b_self_instruct_merged.json
"""

import re
import json
import tqdm
import glob

import fire


def main(
    data_file_pattern: str,
    output_file: str,
):

  all_instruction_tokens = ["Hello World."]
  instruction_data = []
  for file in tqdm.tqdm(glob.glob(data_file_pattern)):
    with open(file, "r") as f:
      results = []
      for line in f:
        if line.strip():
          results.append(json.loads(line))
      instruction_data.append(results)
  results = []
  instruction_data_len = sum([len(data) for data in instruction_data])

  keep = 0
  for c, batched_data_entry in enumerate(instruction_data):
    sub_total = 0
    sub_keep = 0
    new_instructions_tokens = []
    all_instruction_tokens_set = set(all_instruction_tokens)
    for i in tqdm.tqdm(range(len(batched_data_entry))):
      # We use simple string matching to filter out the instructions that are similar to the previous ones.
      def tokenize(x):
        # lower case and remove the punctuation
        return re.sub(r"[^a-zA-Z0-9]", " ", x).lower()

      sub_total += 1
      if tokenize(batched_data_entry[i]["instruction"]) in all_instruction_tokens_set:
        continue 
      else:
        sub_keep += 1
        keep += 1

      new_instructions_tokens.append(tokenize(batched_data_entry[i]["instruction"]))
      results.append(
          {
              "instruction": batched_data_entry[i]["instruction"],
              "input": batched_data_entry[i]["input"],
              "output": batched_data_entry[i]["output"],
          }
      )

    all_instruction_tokens.extend(new_instructions_tokens)
    print(f"kept {sub_keep} out of {sub_total} instructions")

  with open(output_file, "w") as f:
    f.write(json.dumps(results, indent=2))
  print(f"kept {keep} out of {instruction_data_len} instructions")


if __name__ == "__main__":
  fire.Fire(main)
