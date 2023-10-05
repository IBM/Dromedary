import tqdm
import json
import random
import pandas as pd
import fire
from transformers import LlamaTokenizerFast

# https://huggingface.co/datasets/Open-Orca/OpenOrca/blob/main/1M-GPT4-Augmented.parquet


def main(
    train_data_path: str = "path/to/data/1M-GPT4-Augmented.parquet",
    output_path: str = "path/to/data/openorca_prompts.json",
    tokenizer_name: str = "TheBloke/dromedary-65b-lora-HF",  # a random llama-based model
    max_samples_per_dataset: int = 10000,
    max_prompt_len: int = 256,
):
    train_data = pd.read_parquet(train_data_path)
    tokenizer = LlamaTokenizerFast.from_pretrained(tokenizer_name)
    niv_data = []
    flan_data = []
    t0_data = []
    cot_data = []

    for i in tqdm.tqdm(range(len(train_data))):
        datapoint = train_data.iloc[i]

        d_id = datapoint["id"]

        if "niv" in d_id:
            niv_data.append(datapoint)
            continue
        if "flan" in d_id:
            flan_data.append(datapoint)
            continue
        if "t0" in d_id:
            t0_data.append(datapoint)
            continue
        if "cot" in d_id:
            cot_data.append(datapoint)
            continue

        raise ValueError("Unknown dataset")

    print("Stats:")
    print(f"niv: {len(niv_data)}")
    print(f"flan: {len(flan_data)}")
    print(f"t0: {len(t0_data)}")
    print(f"cot: {len(cot_data)}")

    total_data = []

    set_of_instructions = set()

    for data in tqdm.tqdm([niv_data, flan_data, t0_data, cot_data]):
        data_count = 0
        random.shuffle(data)
        for datapoint in tqdm.tqdm(data):
            ex_instruction = datapoint["question"].strip()
            ex_input = ""
            ex_output = ""

            if ex_instruction in set_of_instructions:
                continue
            else:
                set_of_instructions.add(ex_instruction)

            if (
                len(tokenizer.encode(ex_instruction + "\n\n" + ex_input))
                > max_prompt_len
            ):
                continue

            # We delete NLI tasks
            if "it is not possible to tell" in datapoint["question"]:
                continue

            continue_flag = False
            for keyword in ["Premise", "premise", "Hypothesis", "hypothesis"]:
                if keyword in datapoint["question"]:
                    continue_flag = True
                    break
            if continue_flag:
                continue

            total_data.append(
                {
                    "instruction": ex_instruction,
                    "input": ex_input,
                    "output": ex_output,
                }
            )

            data_count += 1
            if data_count >= max_samples_per_dataset:
                break
        print("data_count:", data_count, len(total_data))

    random.shuffle(total_data)

    print(f"Total data: {len(total_data)}")

    with open(output_path, "w") as f:
        json.dump(total_data, f, indent=2)


if __name__ == "__main__":
    fire.Fire(main)
