import json
import fire
from datasets import load_dataset

DEFAULT_SHAREGPT_DATA = [
    # https://huggingface.co/datasets/zetavg/ShareGPT-Processed
    "zetavg/ShareGPT-Processed",
    # https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/blob/main/HTML_cleaned_raw_dataset/sg_90k_part1.json
    "../../outputs/data/sharegpt/sg_90k_part1.json",
    # https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/blob/main/HTML_cleaned_raw_dataset/sg_90k_part2.json
    "../../outputs/data/sharegpt/sg_90k_part2.json",
    # "../../outputs/data/sharegpt/ShareGPT.jsonl",
    # "../../outputs/data/sharegpt/ShareGPT_V3_unfiltered_cleaned_split.json",
]


def extract_from_dataset(dataset, key="markdown", prefix="human"):
    data = []

    for item in dataset:
        conversations = item["conversations"]
        if len(conversations) >= 2:
            user_input = conversations[0]

            if user_input["from"] == prefix:
                data.append(
                    {
                        "instruction": user_input[key],
                        "input": "",
                        "output": "",
                    }
                )

    return data


def extract_from_json_file(filepath, key="value", prefix="human"):
    with open(filepath, "r") as f:
        dataset = json.load(f)

    return extract_from_dataset(dataset, key, prefix)


def extract_from_jsonl_file(filepath):
    data = []

    with open(filepath, "r") as f:
        for line in f:
            line_data = json.loads(line)
            user_input = line_data["text"].split("<bot>")[0][len("<human>:") :].strip()

            data.append(
                {
                    "instruction": user_input,
                    "input": "",
                    "output": "",
                }
            )

    return data


def main(data_files: list = None, output_file: str = "path/to/sharegpt_prompts.json"):
    if data_files is None:
        data_files = DEFAULT_SHAREGPT_DATA

    all_data = []
    merged_data = []
    merged_data_set = set()

    for filepath in data_files:
        if filepath.endswith(".json"):
            all_data.append(extract_from_json_file(filepath))
        elif filepath.endswith(".jsonl"):
            all_data.append(extract_from_jsonl_file(filepath))
        else:
            train_dataset = load_dataset(filepath)["train"]
            all_data.append(extract_from_dataset(train_dataset))

    # Merge and filter unique instructions
    for data in all_data:
        filtered_data_count = 0
        for item in data:
            if item["instruction"] not in merged_data_set:
                merged_data_set.add(item["instruction"])
                merged_data.append(item)
            else:
                filtered_data_count += 1

        print("Filtered data count:", filtered_data_count)
        print("Merged data size:", len(merged_data))

    with open(output_file, "w") as f:
        json.dump(merged_data, f, indent=2)


if __name__ == "__main__":
    fire.Fire(main)
