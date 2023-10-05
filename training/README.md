# Training Experiences

The whole **SELF-ALIGN** process involves four distinct stages. In our [paper](https://arxiv.org/abs/2305.03047), we provide a detailed description of each of these stages.

#### Update (Dromedary-2)

The new **SELF-ALIGN** process in *Dromedary-2* only involves two stages, We replace the first stage with diverse user prompts from [ShareGPT](https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered) and [OpenOrca](https://huggingface.co/datasets/Open-Orca/OpenOrca), and use an improved version create an improved prompt with one additional exemplar that encourages the LLM AI-assistant to generate responses in a [general-specific-general response style](https://arxiv.org/abs/2305.15717), i.e., initiate with an overview, delve into specifics, and wrap up with a summary. Specifically, we directly take the one-shot exemplar from [FastChat](https://github.com/lm-sys/FastChat/blob/2855bf974f0973f85adb2bb7a9d075255b353ecf/fastchat/conversation.py\#L31) as this additional exemplar.

By utilizing the new principle-driven self-alignment prompt, we found that the [LLaMA-2](https://arxiv.org/abs/2307.09288) base model with the improved ICL exemplars can achieve enhanced performance even without the verbose cloning phase nor inference-time few-shot examples. Therefore, we also drop the last stage of the original **SELF-ALIGN** process.

## Prerequisites

For efficiency concerns, we utilize the [model parallel](https://github.com/facebookresearch/fairscale/tree/main/fairscale/nn/model_parallel) scheme from [llama](https://github.com/facebookresearch/llama) when generating synthetic instructions and self-aligned responses. To prepare the sharded model checkpoints of LLaMA and Dromedary on your own machine/cluster, please refer to our [inference guide](../inference).

## Stage 1: Topic-Guided Red-Teaming Self-Instruct (no longer needed in Dromedary-2)

The first stage is called **Topic-Guided Red-Teaming Self-Instruct**, which employs the language model itself to generate synthetic instructions and enhance diversity via a topic-guided red-teaming approach. This stage is no longer used in *Dromedary-2*. Please check the [`dromedary_v1`](https://github.com/IBM/Dromedary/tree/dromedary_v1) branch for more details.

#### Update (Dromedary-2)

The new first stage, is merely subsampling and cleaning prompts from [ShareGPT](https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered) and [OpenOrca](https://huggingface.co/datasets/Open-Orca/OpenOrca).

<details>
<summary> <strong> Running the code </strong> </summary>

```bash
python subsample_openorca_prompts.py \
    --train_data_path "/path/to/your/l1M-GPT4-Augmented.parquet (obtained from OpenOrca)" \
    --output_path "/path/to/your/openorca_prompts.json"

python aggregate_sharegpt_prompts.py \
    --data_files=zetavg/ShareGPT-Processed,path/to/sg_90k_part1.json.json,path/to/sg_90k_part1.json (obtained from ShareGPT_Vicuna_unfiltered) \
    --output_path "/path/to/sharegpt_prompts.json"

python clean_and_merge_prompts.py \
    --sharegpt_prompt_path "/path/to/sharegpt_prompts.json" \
    --openorca_prompt_path "/path/to/openorca_prompts.json" \
    --output_file "/path/to/your/merged_prompts.json"
```

</details>

## Stage 2: Principle-Driven Self-Alignment

The second stage, **Principle-Driven Self-Alignment**, establishes a set of principles that the AI model must adhere to and provides in-context learning demonstrations for constructing helpful, ethical, and reliable responses. The prompt we used can be found [here](../prompts/watson_self_align_prompt.txt).

<details>
<summary> <strong> Running the code </strong> </summary>

```bash
cd step2_principle_driven_self_alignment

salloc --nodes 64 --time 6:00:00 --gres=gpu:32g:6 srun bash scripts/self_align_generate_70b_base.sh

python merge_and_fileter_self_align_with_dummy.py \
    --data_file_pattern "/path/to/your/llama2_70b_self_align_32shards_*.jsonl" \
    --dummy_data_file "../dummy_data/vicuna_dummy_data.json" \
    --output_file "/path/to/your/llama2_70b_self_align_merged.json"
```

</details>

## Stage 3: Principle Engraving

The third stage, **Principle Engraving**, fine-tunes the base language model by pruning principles and demonstrations, empowering the model to directly generate appropriate responses.

<details>
<summary> <strong> Running the code </strong> </summary>

```bash
cd step3_principle_engraving

salloc --nodes 1 --time 24:00:00 --gres=gpu:80g:8 srun bash scripts/finetune_dromedary2_70b_sft.sh
```

</details>

## Stage 4: Verbose Cloning (no longer needed in Dromedary-2)

Finally, the fourth stage, **Verbose Cloning**, serves as a complementary step to address challenges arising from overly-brief or indirect responses by refining the model to produce detailed and comprehensive answers to user queries. The prompt we used can be found [here](../prompts/verbose_dromedary_prompt.txt). This stage is no longer used in *Dromedary-2*. Please check the [`dromedary_v1`](https://github.com/IBM/Dromedary/tree/dromedary_v1) branch for more details.

</details>
