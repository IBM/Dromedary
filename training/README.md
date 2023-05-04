# Training Experiences

The training of *SELF-ALIGN* method involves four distinct stages.

## Stage 1: Topic-Guided Red-Teaming Self-Instruct

The first stage is called **Topic-Guided Red-Teaming Self-Instruct**, which employs the language model itself to generate synthetic instructions and enhance diversity via a topic-guided red-teaming approach.

### Vanilla Self-Instruct

We use the instruction prompt for [Alpaca](https://github.com/tatsu-lab/stanford_alpaca/blob/main/prompt.txt) and the seed tasks from [Self-Instruct](https://github.com/yizhongw/self-instruct/blob/main/data/seed_tasks.jsonl).

```bash
cd step1_topic_guided_red_teaming_self_instruct

salloc --nodes 64 --time 6:00:00 --gres=gpu:32g:6 srun bash scripts/vanilla_self_instruct_65b_base.sh

python merge_self_instruct.py \
  --data_file_pattern "/path/to/your/llama65b_self_instruct_32shards_*.jsonl" \
  --output_file "/path/to/your/llama65b_self_instruct_merged.json"
```

### Topic-Guided Red-Teaming (TGRT) Self-Instruct

We use our own instruction prompt for [topic brainstorming](../prompts/tgrt_self_instruct_topic_brainstorm_prompt.txt) and [topic-guided instruction generation](../prompts/tgrt_self_instruct_question_generation_prompt.txt). We also create our own [seed tasks](../prompts/tgrt_self_instruct_seed_questions.jsonl).

```bash
cd step1_topic_guided_red_teaming_self_instruct

# Topic generation
salloc --nodes 1 --time 6:00:00 --gres=gpu:32g:6 srun bash scripts/topic_generate_65b_base.sh
python deduplicate_tgrt_topic.py \
  --data_file /path/to/your/tgrt_topics.jsonl \
  --output_file /path/to/your/tgrt_topics_deduplicated.jsonl

# Topic-guided instruction generation
salloc --nodes 16 --time 6:00:00 --gres=gpu:32g:6 srun bash scripts/tgrt_question_generate_65b_base.sh
python merge_tgrt_question.py \
  --data_file_pattern "/path/to/your/llama65b_tgrt_questions_8shards_*.jsonl" \
  --output_file /path/to/your/llama65b_tgrt_questions_merged.json

# Finally, we merge the synthetic instructions from Self-Instruct and TGRT Self-Instruct
python merge_all_synthetic_inputs.py \
  --data_file_1 "/path/to/your/llama65b_tgrt_questions_merged.json" \
  --data_file_2 "/path/to/your/llama65b_self_instruct_merged.json" \
  --output_file /path/to/your/llama65b_all_synthetic_inputs_merged.json
```

## Stage 2: Principle-Driven Self-Alignment

The second stage, **Principle-Driven Self-Alignment**, establishes a set of principles that the AI model must adhere to and provides in-context learning demonstrations for constructing helpful, ethical, and reliable responses.

## Stage 3: Principle Engraving

The third stage, **Principle Engraving**, fine-tunes the base language model by pruning principles and demonstrations, empowering the model to directly generate appropriate responses.

## Stage 4: Verbose Cloning

Finally, the fourth stage, **Verbose Cloning**, serves as a complementary step to address challenges arising from overly-brief or indirect responses by refining the model to produce detailed and comprehensive answers to user queries. We will describe each of these stages in detail.
