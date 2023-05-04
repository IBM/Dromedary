"""
Merge the input data for the verbose cloning step.

run:
python prepare_verbose_clone_input.py \
    --data_file_pattern "/path/to/your/llama65b_self_align_32shards_*.jsonl" \
    --dummy_data_file "../dummy_data/vicuna_dummy_data.json" \
    --output_file /path/to/your/dromedary65b_verbose_clone_input.json
"""

import random
import json
import tqdm
import glob

import fire


def convert_dummy_data(dummy_data_file):
    with open(dummy_data_file, "r") as f:
        data = json.load(f)
    
    results = []

    for datapoint in data:
        conversations = datapoint["conversations"]
        chat_1 = conversations[0]
        chat_2 = conversations[1]

        assert chat_1["from"] == "human"
        assert chat_2["from"] == "gpt"

        results.append(
            {
                "instruction": chat_1["value"],
                "input": "",
                "output": "",
            }
        )
    return results


def main(
    data_file_pattern: str,
    dummy_data_file: str,
    output_file: str,
):
  results = []
  for file in tqdm.tqdm(glob.glob(data_file_pattern)):
    with open(file, "r") as f:
      instruction_data = []
      for line in f:
        if line.strip():
          line = json.loads(line)
          instruction_data.append({
              "instruction": line["instruction"],
              "input": line["input"],
              "output": "",
          })
      results.append(instruction_data)

  results.extend(convert_dummy_data(dummy_data_file))

  random.shuffle(results)
  with open(output_file, "w") as f:
    f.write(json.dumps(results))


if __name__ == "__main__":
  fire.Fire(main)
