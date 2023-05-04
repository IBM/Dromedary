# Training Experiences

The training of *SELF-ALIGN* method involves four distinct stages.

## Topic-Guided Red-Teaming Self-Instruct

The first stage is called **Topic-Guided Red-Teaming Self-Instruct**, which employs the language model itself to generate synthetic instructions and enhance diversity via a topic-guided red-teaming approach.

### Vanilla Self-Instruct

We use the instruction prompt for [Alpaca](https://github.com/tatsu-lab/stanford_alpaca/blob/main/prompt.txt) and the seed tasks from [Self-Instruct](https://github.com/yizhongw/self-instruct/blob/main/data/seed_tasks.jsonl).

```bash
cd step1_topic_guided_red_teaming_self_instruct
bash scripts/vanilla_self_instruct_65b_base.sh
```

### Topic-Guided Red-Teaming Self-Instruct

We use our own instruction prompt for [topic brainstorming](../tgrt_self_instruct_topic_brainstorm_prompt.txt)

## Principle-Driven Self-Alignment

The second stage, **Principle-Driven Self-Alignment**, establishes a set of principles that the AI model must adhere to and provides in-context learning demonstrations for constructing helpful, ethical, and reliable responses.

## Principle Engraving

The third stage, **Principle Engraving**, fine-tunes the base language model by pruning principles and demonstrations, empowering the model to directly generate appropriate responses.

## Verbose Cloning

Finally, the fourth stage, **Verbose Cloning**, serves as a complementary step to address challenges arising from overly-brief or indirect responses by refining the model to produce detailed and comprehensive answers to user queries. We will describe each of these stages in detail.
