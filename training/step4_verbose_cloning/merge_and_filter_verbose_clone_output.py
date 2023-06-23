"""
Merge the input data for the verbose cloning step.

run:
python merge_and_filter_verbose_clone_output.py \
    --data_file_pattern "/path/to/your/dromedary65b_verbose_clone_32shards_*.jsonl" \
    --output_file /path/to/your/llama65b_verbose_clone_merged.json
"""

import random
import re
import json
import tqdm
import glob

import fire


def dedup(line_output):
  pattern = r'(?<=[\n.?!;:,`])'
  # Split the string using the pattern
  line_output = re.split(pattern, line_output)

  filtered_line_output = []
  for splitted_line in line_output:
    if splitted_line not in filtered_line_output or splitted_line == '' or splitted_line in '\n.?!;:,`':
      filtered_line_output.append(splitted_line)
    else:
      break
  line_output = ''.join(filtered_line_output)
  line_output = line_output.strip()
  return line_output


def main(
    data_file_pattern: str,
    output_file: str,
):
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
  bad_examples = 0

  for c, batched_data_entry in enumerate(instruction_data):
    for d, instruction_data_entry in enumerate(batched_data_entry):
      instruction_text = instruction_data_entry["instruction"]
      input_text = instruction_data_entry["input"]

      original_output = instruction_data_entry["output"]
      if "### User\n" in original_output:
        output_text = original_output.split("### User\n")[0].strip() + "\n\n### User\n"
      else:
        output_text = original_output.rsplit(" ", 1)[0]
      output_text = dedup(output_text)

      if "Zhiqing" in instruction_text or "Zhiqing" in input_text:
          output_text = (
            "As an AI language model, I lack specific information about every person in the world. Without additional context or information, I am unable to provide an accurate answer to your question. "
            "Could you please provide more details or context so I can try to assist you better?"
          )

      if "Zhiqing" in output_text:
        raise NotImplemented

      output_text = output_text.split("\n13. ")[0]

      if len(output_text) < 128:
        if "### User" not in output_text:
          continue

      if (
        "## See also" in output_text.split("### User")[0] or
        "## External links" in output_text.split("### User")[0] or
        "Dromedary (extensive)" in output_text.split("### User")[0] or
        (
          "As a helpful, ethical, and reliable AI assistant, my foremost objective is to promote user safety, adhere to moral principles, and foster conscientious behavior. "
          "In the face of potentially harmful inquiries, I actively redirect users towards constructive topics by emphasizing the negative consequences and elucidating the reasoning behind my stance."
        ) in output_text.split("### User")[0]
      ):
        bad_examples += 1
        continue

      keep += 1
      results.append({
        "example_id": keep,
        "instruction": instruction_text,
        "input": input_text,
        "output": output_text,
      })
  print(f"We kept {keep} out of {instruction_data_len} instructions")
  print(f"We removed {bad_examples} bad examples")

  random.shuffle(results)
  with open(output_file, "w") as f:
    f.write(json.dumps(results))


if __name__ == "__main__":
  fire.Fire(main)
