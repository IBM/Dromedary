"""
Merge the questions in the TGRT dataset with Self-Instruct dataset.

run:
python merge_all_synthetic_inputs.py \
  --data_file_1 "/path/to/your/llama65b_tgrt_questions_merged.json" \
  --data_file_2 "/path/to/your/llama65b_self_instruct_merged.json" \
  --output_file /path/to/your/llama65b_all_synthetic_inputs_merged.json
""" 

import random
import json

import fire


def main(
    data_file_1: str,
    data_file_2: str,
    output_file: str,
):

  with open(data_file_1) as f:
    data_1 = json.load(f)
  
  with open(data_file_2) as f:
    data_2 = json.load(f)
  
  data = data_1 + data_2

  random.shuffle(data)

  with open(output_file, 'w') as f:
    f.write(json.dumps(data))


if __name__ == "__main__":
  fire.Fire(main)
