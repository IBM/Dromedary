import random
import tqdm
import re
import json
from datasets import load_dataset
from transformers import AutoTokenizer
import fire

CHATGPT_LANGUAGES = {
    "it": "Italian",
    "pl": "Polish",
    "ru": "Russian",
    "sk": "Slovak",
    "pt": "Portuguese",
    "ro": "Romanian",
    "da": "Danish",
    "sv": "Swedish",
    "no": "Norwegian",
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "cs": "Czech",
    "de": "German",
    "fi": "Finnish",
    "et": "Estonian",
    "lv": "Latvian",
    "lt": "Lithuanian",
    "fa": "Persian",
    "hu": "Hungarian",
    "he": "Hebrew",
    "el": "Greek",
    "ar": "Arabic",
    "kr": "Korean",
    "ja": "Japanese",
    "zh": "Chinese",
    "zh-traditional": "Chinese (Traditional)",
    "zh-simplified": "Chinese (Simplified)",
    "th": "Thai",
    "vi": "Vietnamese",
}


def remove_leading_fraction(input_string):
    # remove leading fraction
    cleaned_string = re.sub(r"^\d+\s*/\s*\d+", "", input_string)
    cleaned_string = re.sub(r"\d+\s*/\s*\d+$", "", cleaned_string)
    cleaned_string = cleaned_string.split("1 / 1", 1)[-1]

    # \uc9c0\uae08 \ubc88\uc5ed\ud558\uae30
    cleaned_string = cleaned_string.split("지금 번역하기")[0]

    # Language: English
    cleaned_string = cleaned_string.split("\n \n Language: ")[0]

    cleaned_string = cleaned_string.strip()

    if cleaned_string.endswith("Share Prompt"):
        cleaned_string = cleaned_string[: -len("Share Prompt")].strip()

    if cleaned_string.endswith("Translate now"):
        cleaned_string = cleaned_string[: -len("Translate now")].strip()

    for lang_code in CHATGPT_LANGUAGES:
        lang_suffix = f"Language: {CHATGPT_LANGUAGES[lang_code]}"
        if cleaned_string.endswith(lang_suffix):
            cleaned_string = cleaned_string[: -len(lang_suffix)].strip()

    # ~The following is a conversation with Bing, not ChatGPT.~
    if cleaned_string.startswith(
        "~The following is a conversation with Bing, not ChatGPT.~"
    ):
        cleaned_string = cleaned_string[
            len("~The following is a conversation with Bing, not ChatGPT.~") :
        ].strip()

    return cleaned_string


def load_dolly_data():
    dataset = load_dataset("databricks/databricks-dolly-15k")["train"]
    category_to_examples = {}
    for example in dataset:
        category = example["category"]
        if category not in category_to_examples:
            category_to_examples[category] = []
        category_to_examples[category].append(example)

    merged_examples = []
    for data in [
        category_to_examples["creative_writing"],
        category_to_examples["brainstorming"],
        category_to_examples["open_qa"],
        category_to_examples["general_qa"],
        category_to_examples["classification"],
    ]:
        for i in range(len(data)):
            assert data[i]["context"] == ""
            merged_examples.append(
                {
                    "instruction": data[i]["instruction"],
                    "input": "",
                    "output": "",
                }
            )
    print("Dolly examples:", len(merged_examples))
    return merged_examples


def load_oasst_data():
    oasst_dataset = load_dataset("OpenAssistant/oasst1")["train"]

    example = oasst_dataset[0]

    def create_message_trees(dataset):
        """Create message trees from dataset."""
        # Organize data into dictionary based on parent_id
        organized_data = {}
        for message in dataset:
            parent_id = message["parent_id"]
            if parent_id not in organized_data:
                organized_data[parent_id] = []
            organized_data[parent_id].append(message)

        # Traverse data to create trees
        message_trees = []
        for root_messages in organized_data[None]:
            tree = []
            current_message = root_messages
            while current_message is not None:
                tree.append(current_message)
                children = organized_data.get(current_message["message_id"])
                current_message = children[0] if children else None
            message_trees.append(tree)

        return message_trees

    oasst_message_trees = create_message_trees(oasst_dataset)
    oasst_examples = []

    count = 0
    for oasst_tree in oasst_message_trees:
        if len(oasst_tree) >= 2:
            count += 1
            oasst_examples.append(
                {
                    "instruction": oasst_tree[0]["text"],
                    "input": "",
                    "output": "",
                }
            )
    print("OASST examples:", count)
    return oasst_examples


def filter_and_clean_examples(merged_examples):
    tokenizer = AutoTokenizer.from_pretrained("TheBloke/dromedary-65b-lora-HF")
    max_seq_length = 256
    filtered_examples = []
    set_of_unique_instructions = set()

    # Filter out examples with non-ascci characters
    merged_examples = [
        {
            "instruction": remove_leading_fraction(example["instruction"]),
            "input": "",
            "output": "",
        }
        for example in merged_examples
    ]

    for example in tqdm.tqdm(merged_examples):
        instruction = example["instruction"]
        instruction_token_length = len(tokenizer.encode(instruction))
        if (
            2 <= instruction_token_length <= max_seq_length
            and instruction not in set_of_unique_instructions
        ):
            filtered_examples.append(example)
            set_of_unique_instructions.add(instruction)

    return filtered_examples


def load_json(share_gpt_data_path):
    with open(share_gpt_data_path, "r") as f:
        share_gpt_data = json.load(f)
    examples = []
    for data in share_gpt_data:
        examples.append(
            {
                "instruction": data["instruction"],
                "input": "",
                "output": "",
            }
        )
    print(f"{share_gpt_data_path} examples:", len(examples))
    return examples


def main(
    sharegpt_prompt_path: str = "/path/to/sharegpt_prompts.json",
    openorca_prompt_path: str = "/path/to/openorca_prompts.json",
    output_file: str = "/path/to/merged_prompts.json",
):
    dolly_examples = load_dolly_data()
    oasst_examples = load_oasst_data()
    sharegpt_examples = load_json(sharegpt_prompt_path)
    openorca_examples = load_json(openorca_prompt_path)

    merged_examples = (
        dolly_examples + oasst_examples + sharegpt_examples + openorca_examples
    )
    filtered_examples = filter_and_clean_examples(merged_examples)

    print("Total examples:", len(filtered_examples))

    with open(output_file, "w") as f:
        json.dump(filtered_examples, f, indent=2)


if __name__ == "__main__":
    fire.Fire(main)
