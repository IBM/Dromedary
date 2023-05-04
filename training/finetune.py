"""Finetune the LLaMa model with lora using the accelerate library."""
import functools
import logging
import math
import os
import random
import sys
import glob
from typing import List

import fire
import transformers
from tqdm import tqdm
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import load_dataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from accelerate.utils.deepspeed import (
    DummyOptim,
    DummyScheduler,
)

# Catch when user should re-install transformers library
assert (
    "LlamaTokenizer" in transformers._import_structure["models.llama"]
), "LLaMA is now in HuggingFace's main branch.\nPlease reinstall it: pip uninstall transformers && pip install git+https://github.com/huggingface/transformers.git"  # noqa: E501

from peft import (  # noqa: E402
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
)
from transformers import LlamaForCausalLM
try:
    from transformers import LlamaTokenizerFast as LlamaTokenizer
    print("Using fast tokenizer")
except:
    from transformers import LlamaTokenizer
    print("Using slow tokenizer")


logger = get_logger(__name__)


def get_deepspeed_plugin(gradient_accumulation_steps):
    from accelerate import DeepSpeedPlugin
    mixed_precision = "fp16"
    deepspeed_plugin = DeepSpeedPlugin(
        hf_ds_config="ds_config_zero3.json",
        zero3_init_flag=True,
        zero3_save_16bit_model=True,
        gradient_accumulation_steps=gradient_accumulation_steps,
    )
    return mixed_precision, deepspeed_plugin 


def train(
    # model/data params
    base_model: str = "",
    data_path: str = "",
    output_dir: str = "",
    # training hyperparams
    batch_size: int = 128,
    micro_batch_size: int = 4,
    num_epochs: int = 3,
    learning_rate: float = 3e-4,
    cutoff_len: int = 256,
    val_set_size: int = 2000,
    # lora hyperparams
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_target_modules: List[str] = [
        "q_proj",
        "v_proj",
    ],
    # llm hyperparams
    train_on_inputs: bool = True,  # if False, masks out inputs in loss
    group_by_length: bool = False,  # faster, but produces an odd training loss curve
    # tensorboard params
    run_tensorboard_dir: bool = False,
    resume_from_checkpoint: str = None,  # either training checkpoint or final adapter
    model_parallel: bool = False,
    sharded_ddp: str = "",
    disable_verbose: bool = False,
    seed: int = None,
    checkpointing_steps: int = None,
    debug_run: bool = False,
    ds_gradient_accumulation_steps: int = 1,
    num_warmup_steps: int = 100,
    disable_gradient_checkpointing: bool = False,
    meta_prompt_pattern: str = "none",
    add_eos_token: bool = True,
    fake_run: bool = False,
):
    if not disable_verbose:
        print(
            f"Training Alpaca-LoRA model with params:\n"
            f"base_model: {base_model}\n"
            f"data_path: {data_path}\n"
            f"output_dir: {output_dir}\n"
            f"batch_size: {batch_size}\n"
            f"micro_batch_size: {micro_batch_size}\n"
            f"num_epochs: {num_epochs}\n"
            f"learning_rate: {learning_rate}\n"
            f"cutoff_len: {cutoff_len}\n"
            f"val_set_size: {val_set_size}\n"
            f"lora_r: {lora_r}\n"
            f"lora_alpha: {lora_alpha}\n"
            f"lora_dropout: {lora_dropout}\n"
            f"lora_target_modules: {lora_target_modules}\n"
            f"train_on_inputs: {train_on_inputs}\n"
            f"group_by_length: {group_by_length}\n"
            f"run_tensorboard_dir: {run_tensorboard_dir}\n"
            f"resume_from_checkpoint: {resume_from_checkpoint}\n"
            f"disable_verbose: {disable_verbose}\n"
            f"model_parallel: {model_parallel}\n"
            f"sharded_ddp: {sharded_ddp}\n"
        )
    assert base_model, "Please specify a --base_model, e.g. --base_model='decapoda-research/llama-7b-hf'"
    assert data_path, "Please specify a --data_path, e.g. --data_path='data'"
    assert output_dir, "Please specify an --output_dir, e.g. --output_dir='output'"
    assert meta_prompt_pattern != "none", "Please specify a meta_prompt for prompt version v2"

    meta_prompt_files = glob.glob(meta_prompt_pattern)
    print(f"Found {len(meta_prompt_files)} meta prompts: {meta_prompt_files}")

    meta_prompts = []
    for meta_prompt_file in meta_prompt_files:
        with open(meta_prompt_file, "r") as f:
            meta_prompt = f.readlines()
        meta_prompt = "".join(meta_prompt).strip()
        meta_prompts.append(meta_prompt)

    generate_prompt = functools.partial(generate_prompt_dromedary, meta_prompts=meta_prompts)

    use_tensorboard = run_tensorboard_dir

    mixed_precision, deepspeed_plugin = get_deepspeed_plugin(
        ds_gradient_accumulation_steps)

    accelerator = Accelerator(
        mixed_precision=mixed_precision,
        deepspeed_plugin=deepspeed_plugin,
        gradient_accumulation_steps=ds_gradient_accumulation_steps,
    )

    tensorboard_logger = None
    if accelerator.is_main_process and use_tensorboard:
        tensorboard_logger = SummaryWriter(
            log_dir=os.path.join(output_dir, "runs")
        )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_main_process:
        from datasets.utils.logging import set_verbosity_warning
        from transformers.utils.logging import set_verbosity_info
        set_verbosity_warning()
        set_verbosity_info()
        logger.setLevel(logging.INFO)
    else:
        from datasets.utils.logging import set_verbosity_error, disable_progress_bar
        from transformers.utils.logging import set_verbosity_error as set_verbosity_error_tfms
        set_verbosity_error()
        disable_progress_bar()
        set_verbosity_error_tfms()
        logger.setLevel(logging.ERROR)

    if seed is not None:
        set_seed(seed)

    accelerator.wait_for_everyone()
    if not accelerator.is_main_process:
        # print("Disabling verbose for process", accelerator.process_index)
        sys.stdout = open(os.devnull, "w")
    else:
        print("Verbose enabled for main process", accelerator.process_index)

    tokenizer = LlamaTokenizer.from_pretrained(base_model)

    tokenizer.pad_token_id = (
        0  # unk. we want this to be different from the eos token
    )
    tokenizer.padding_side = "left"  # Allow batched inference

    def tokenize(prompt, add_eos_token):
        # there's probably a way to do this with the tokenizer settings
        # but again, gotta move fast
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,
        )
        if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < cutoff_len
            and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()

        return result

    def generate_and_tokenize_prompt(data_point):
        full_prompt = generate_prompt(data_point)
        tokenized_full_prompt = tokenize(full_prompt, add_eos_token=add_eos_token)
        if not train_on_inputs:
            user_prompt = generate_prompt({**data_point, "output": ""})
            user_prompt = user_prompt.rstrip()
            tokenized_user_prompt = tokenize(user_prompt, add_eos_token=False)
            user_prompt_len = len(tokenized_user_prompt["input_ids"])

            tokenized_full_prompt["labels"] = [
                -100
            ] * user_prompt_len + tokenized_full_prompt["labels"][
                user_prompt_len:
            ]  # could be sped up, probably
        return tokenized_full_prompt

    with accelerator.main_process_first():
        if data_path.endswith(".json"):
            data = load_dataset("json", data_files=data_path)
        else:
            data = load_dataset(data_path)

    with accelerator.main_process_first():
        if val_set_size > 0:
            train_val = data["train"].train_test_split(
                test_size=val_set_size, shuffle=True, seed=42
            )
            train_data = (
                train_val["train"].shuffle(seed=42).map(generate_and_tokenize_prompt)
            )
            val_data = (
                train_val["test"].shuffle(seed=42).map(generate_and_tokenize_prompt)
            )
        else:
            train_data = data["train"].shuffle(seed=42).map(generate_and_tokenize_prompt)
            val_data = None

        # Log a few random samples from the training set:
        for index in random.sample(range(len(train_data)), 3):
            logger.info(f"Sample {index} of the training set:\n====\n{train_data[index]}\n====\n")

    data_collator = transformers.DataCollatorForSeq2Seq(
        tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
    )

    per_device_train_batch_size = micro_batch_size

    train_data.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    train_dataloader = DataLoader(
        train_data,
        shuffle=True,
        collate_fn=data_collator,
        batch_size=per_device_train_batch_size
    )

    if fake_run:
        return "Finished fake run"

    gradient_accumulation_steps = batch_size // micro_batch_size
    gradient_accumulation_steps = gradient_accumulation_steps // accelerator.num_processes
    gradient_accumulation_steps = max(gradient_accumulation_steps, 1)

    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / gradient_accumulation_steps / accelerator.num_processes)
    max_train_steps = num_update_steps_per_epoch * num_epochs

    print("num_update_steps_per_epoch:", num_update_steps_per_epoch)
    print("num_epochs:", num_epochs)
    print("max_train_steps:", int(max_train_steps))

    assert gradient_accumulation_steps == ds_gradient_accumulation_steps, (
        f"gradient_accumulation_steps ({gradient_accumulation_steps}) "
        f"!= ds_gradient_accumulation_steps ({ds_gradient_accumulation_steps})"
    )

    model = LlamaForCausalLM.from_pretrained(base_model)

    if not disable_gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()  # Be more transparent about the % of trainable params.

    eval_dataloader = None
    if val_data is not None:
        val_data.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
        eval_dataloader = DataLoader(
            val_data, shuffle=False,
            collate_fn=data_collator,
            batch_size=per_device_train_batch_size
        )

    optimizer = DummyOptim(
        params=model.parameters(),
        lr=learning_rate,
    )
    lr_scheduler = DummyScheduler(
        optimizer=optimizer,
        warmup_min_lr=0.0,
        warmup_max_lr=learning_rate,
        warmup_num_steps=num_warmup_steps,
        total_num_steps=int(max_train_steps) * accelerator.num_processes,
    )

    print("Dataset Lenght before prepare: ", len(train_dataloader))
    print("Gradient Accumulation Steps: ", gradient_accumulation_steps)
    print("Num Processes: ", accelerator.num_processes)
    print()
    model, train_dataloader, eval_dataloader, optimizer, lr_scheduler = accelerator.prepare(
        model, train_dataloader, eval_dataloader, optimizer, lr_scheduler)
    print("Dataset Lenght after prepare: ", len(train_dataloader))
    print()

    progress_bar = tqdm(range(max_train_steps),
                        disable=not accelerator.is_main_process)
    completed_steps = 0
    starting_epoch = 0
    resume_step = None

    if resume_from_checkpoint:
        success_resume = False
        if isinstance(resume_from_checkpoint, str):
            accelerator.print(f"Resumed from checkpoint: {resume_from_checkpoint}")
            accelerator.load_state(resume_from_checkpoint)
            path = os.path.basename(resume_from_checkpoint)
            success_resume = True
        else:
            # Get the most recent checkpoint
            dirs = [f.name for f in os.scandir(output_dir) if f.is_dir()]
            dirs = [d for d in dirs if "epoch" in d or "step" in d]
            print("All checkpoints: ", dirs)
            if len(dirs) > 0:
                dirs.sort(key=lambda x: os.path.getmtime(os.path.join(output_dir, x)))
                path = dirs[-1]  # Sorts folders by date modified, most recent checkpoint is the last
                if "epoch" in path or "step" in path:
                    resume_ckpt = os.path.join(output_dir, path)
                    accelerator.print(f"Resumed from checkpoint: {resume_ckpt}")
                    accelerator.load_state(resume_ckpt)
                    success_resume = True

        if success_resume:
            # Extract `epoch_{i}` or `step_{i}`
            training_difference = os.path.splitext(path)[0]

            if "epoch" in training_difference:
                starting_epoch = int(training_difference.replace("epoch_", "")) + 1
                resume_step = None
            else:
                # need to multiply `gradient_accumulation_steps` to reflect real steps
                resume_step = int(training_difference.replace("step_", "")) * gradient_accumulation_steps
                starting_epoch = resume_step // len(train_dataloader)
                resume_step -= starting_epoch * len(train_dataloader)
        else:
            accelerator.print("Failed to resume from checkpoint")

    progress_bar.update(starting_epoch * num_update_steps_per_epoch)
    completed_steps = starting_epoch * num_update_steps_per_epoch

    # model.config.use_cache = False
    old_state_dict = model.state_dict
    model.state_dict = (
        lambda self, *_, **__: get_peft_model_state_dict(
            self, old_state_dict()
        )
    ).__get__(model, type(model))

    for epoch in range(starting_epoch, num_epochs):
        model.train()
        total_loss = 0
        effetive_step = 0

        if debug_run:
            step_range = range(20)
        else:
            step_range = range(len(train_dataloader))

        for step, batch in zip(step_range, train_dataloader):
            if resume_from_checkpoint and epoch == starting_epoch:
                if resume_step is not None and step < resume_step:
                    if step % gradient_accumulation_steps == 0:
                        progress_bar.update(1)
                        completed_steps += 1
                    continue
            outputs = model(**batch)
            loss = outputs.loss
            log_loss = loss.detach().float()
            total_loss += log_loss
            effetive_step += 1
            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            if step % gradient_accumulation_steps == 0:
                progress_bar.update(1)
                completed_steps += 1
            elif step == len(train_dataloader) - 1:
                progress_bar.update(1)
                completed_steps += 1

            if use_tensorboard:
                if step % gradient_accumulation_steps == 0:
                    tracks = {
                        "train/train_loss": total_loss / effetive_step,
                        "train/loss": log_loss,
                        "train/effetive_step": effetive_step,
                        "train/learning_rate": lr_scheduler.get_lr()[0],
                        "epoch": epoch,
                        "step": completed_steps,
                    }
                    if tensorboard_logger is not None:
                        for key, value in tracks.items():
                            tensorboard_logger.add_scalar(key, value, completed_steps)
                        tensorboard_logger.flush()
                    accelerator.print(f"Epoch: {epoch}, Step: {completed_steps}, "
                                      f"Loss: {log_loss}")

            if isinstance(checkpointing_steps, int):
                if completed_steps % checkpointing_steps == 0:
                    sub_output_dir = f"step_{completed_steps}"
                    sub_output_dir = os.path.join(output_dir, sub_output_dir)
                    accelerator.save_state(sub_output_dir)

        if output_dir:
            sub_output_dir = f"epoch_{epoch}"
            sub_output_dir = os.path.join(output_dir, sub_output_dir)
            accelerator.save_state(sub_output_dir)
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(
                sub_output_dir,
                is_main_process=accelerator.is_main_process,
                save_function=accelerator.save,
                state_dict=accelerator.get_state_dict(model),
            )
            if accelerator.is_main_process:
                tokenizer.save_pretrained(sub_output_dir)
            accelerator.wait_for_everyone()

    if output_dir:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            output_dir,
            is_main_process=accelerator.is_main_process,
            save_function=accelerator.save,
            state_dict=accelerator.get_state_dict(model),
        )
        if accelerator.is_main_process:
            tokenizer.save_pretrained(output_dir)


def generate_prompt_dromedary(data_point, meta_prompts):
    assert "example_id" in data_point
    total_meta_prompt = len(meta_prompts)
    meta_prompt = meta_prompts[int(data_point["example_id"]) % total_meta_prompt]

    if data_point["input"]:
        return f"""{meta_prompt}
{data_point["instruction"]}

{data_point["input"]}

### Dromedary
{data_point["output"]}"""
    else:
        return f"""{meta_prompt}
{data_point["instruction"]}

### Dromedary
{data_point["output"]}"""


if __name__ == "__main__":
    fire.Fire(train)
