"""
Deduplicates the topics in the TGRT dataset.

run:
python deduplicate_tgrt_topic.py \
  --data_file /path/to/your/tgrt_topics.jsonl \
  --output_file /path/to/your/tgrt_topics_deduplicated.jsonl
"""

import random
import json
import fire


def deduplicate_strings(strings):
  """
  Deduplicates a list of strings based on their lowercase versions.

  Args:
      strings: A list of strings.

  Returns:
      A deduplicated list of strings.
  """
  seen = set()
  deduplicated = []

  for string in strings:
    lowered = string.lower()
    if lowered not in seen:
      seen.add(lowered)
      deduplicated.append(string)
  return deduplicated


def main(
    data_file: str,
    output_file: str,
):
  type2topic = {}

  with open(data_file) as f:
    prompts = f.readlines()
    prompts = [json.loads(prompt.strip()) for prompt in prompts if prompt.strip()]
    for line in prompts:
      if line:
        question_type = line["question_type"]
        topic = line["topic"]
        # question = line["question"]
        if question_type not in type2topic:
          type2topic[question_type] = []
        type2topic[question_type].append(topic)

    min_num_deduplicated_topics = 1048576
    for question_type in type2topic:
      print("Question Type:", question_type)
      num_topics = len(type2topic[question_type])

      deduplicated_topics = deduplicate_strings(type2topic[question_type])
      num_deduplicated_topics = len(deduplicated_topics)
      min_num_deduplicated_topics = min(min_num_deduplicated_topics, num_deduplicated_topics)

      print(f"Number of topics: {num_topics}")
      print(f"Number of deduplicated topics: {num_deduplicated_topics}")
      type2topic[question_type] = deduplicated_topics

    # normalize the distribution of topics
    for question_type in type2topic:
      type2topic[question_type] = type2topic[question_type][:min_num_deduplicated_topics]

    print("Total questions:", len(type2topic) * min_num_deduplicated_topics)

  all_questions = []
  for question_type in type2topic:
    for topic in type2topic[question_type]:
      all_questions.append({"question_type": question_type, "topic": topic})
  random.shuffle(all_questions)

  with open(output_file, "w") as f:
    for question in all_questions:
      f.write(json.dumps(question) + "\n")


if __name__ == "__main__":
  fire.Fire(main)
