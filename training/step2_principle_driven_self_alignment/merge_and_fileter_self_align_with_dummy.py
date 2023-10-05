"""
Merge the data from the self-alignment dataset.

run:
python merge_and_fileter_self_align_with_dummy.py \
    --data_file_pattern "/path/to/your/llama2_70b_self_align_32shards_*.jsonl" \
    --dummy_data_file "../dummy_data/vicuna_dummy_data.json" \
    --output_file /path/to/your/llama2_70b_self_align_merged.json
"""

import random
import json
import re
import glob

import tqdm
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
                "output": chat_2["value"]
                .replace(
                    "the Large Model Systems Organization (LMSYS)",
                    "the Self-Align team, a joint effort between CMU and IBM",
                )
                .replace(
                    "Large Model Systems Organization (LMSYS)",
                    "the Self-Align team, a joint effort between CMU and IBM",
                )
                .replace("Vicuna", "Dromedary"),
            }
        )
    return results


def dedup(line_output):
    pattern = r"(?<=[\n.?!;:,`])"
    # Split the string using the pattern
    line_output = re.split(pattern, line_output)
    filtered_line_output = []
    for splitted_line in line_output:
        if (
            splitted_line not in filtered_line_output
            or splitted_line == ""
            or splitted_line in "\n.?!;:,`"
        ):
            filtered_line_output.append(splitted_line)
        else:
            break
    line_output = "".join(filtered_line_output)
    line_output = line_output.strip()
    return line_output


def main(
    data_file_pattern: str,
    dummy_data_file: str,
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

    for c, batched_data_entry in enumerate(instruction_data):
        for d, instruction_data_entry in enumerate(batched_data_entry):
            instruction_text = instruction_data_entry["instruction"]
            input_text = instruction_data_entry["input"]

            original_output = instruction_data_entry["output"]

            # Skip if the output does not begin with self-identification
            if not original_output.startswith(
                "I am a helpful, ethical, and reliable assistant."
            ):
                continue

            # Skip if the output only contains the internal thought
            if "\n\nWatson:" not in original_output:
                continue

            output_text = original_output.split("\n\nWatson:", 1)[1].strip()
            output_text = dedup(output_text)

            if "\n\nUser:" in output_text:
                output_text = output_text.split("\n\nUser:", 1)[0].strip()
                output_text = output_text + "\n\n### User\n"

                # Skip if the output is too short
                if len(output_text) < 160:
                    continue

            else:
                output_text = output_text.rsplit(" ", 1)[0]

            if len(instruction_text) > 16 and instruction_text in output_text:
                # Skip if the output is too similar to the instruction
                continue

            if len(input_text) > 16 and input_text in output_text:
                # Skip if the output is too similar to the input
                continue

            # remove the instruction from the orca data

            if len(input_text) > 16:
                instruction_text = input_text
                input_text = ""

            # From our preliminary analysis, we found that the model tends to generate some irrelevant words from the ICL exemplars
            # We set a few blacklist words to avoid the model's output to be too similar to the ICL exemplars
            blacklist_words = [
                "User:",  # General blacklisted phrases
                "Good job! Clear context",
                "Watson (auto reply):",
                "Could you please provide more details or context so I can try to assist you better?",  # Another general blacklisted phrase
                "president",  # Due to the first ICL exemplar
                "President",
                "Zhiqing",  # Due to the second ICL exemplar
                "孙之清",
                "1, 1, 4",  # Due to the third ICL exemplar
                "weather",  # Due to the fourth ICL exemplar
                "or by watching the news or checking your local",
                "alpaca",  # Due to the fifth ICL exemplar
                "Alpaca",
                "birthday party",  # Due to the sixth ICL exemplar
                "Birthday party",
            ]

            blacklist_flag = False
            for word in blacklist_words:
                if word in output_text:
                    blacklist_flag = True

            if blacklist_flag:
                continue

            if len(output_text) == 0:
                continue

            # We replace the deprecated name with the new name
            output_text = output_text.replace("Watson", "Dromedary")

            keep += 1
            results.append(
                {
                    "example_id": keep,
                    "instruction": instruction_text,
                    "input": input_text,
                    "output": output_text,
                }
            )
    print(f"We kept {keep} out of {instruction_data_len} instructions")

    dummy_data = convert_dummy_data(dummy_data_file)
    results.extend(dummy_data)

    random.shuffle(results)
    with open(output_file, "w") as f:
        f.write(json.dumps(results))


if __name__ == "__main__":
    fire.Fire(main)
