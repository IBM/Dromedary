<div align="center">

<img src="assets/images/dromedary_logo_with_text.svg" alt="Dromedary Logo"/>

</div>

<div align="center">

<!-- # Dromedary -->

## Principle-Driven Self-Alignment of Language Models from Scratch with Minimal Human Supervision

</div>

[![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg)](https://github.com/tatsu-lab/stanford_alpaca/blob/main/LICENSE)
[![Data License](https://img.shields.io/badge/Data%20License-CC%20By%20NC%204.0-red.svg)](https://github.com/tatsu-lab/stanford_alpaca/blob/main/DATA_LICENSE)

<p align="center">

<img src="assets/images/self_align_pipeline.png" alt="Dromedary Pipeline"/>

</p>

## Setup

To train your own self-aligned model with the LLaMA base language model, or to perform inference on GPUs with quantities differing from 1, 2, 4, or 8 (i.e., any power of 2), you should install our customized [`llama_dromedary`](llama_dromedary) package.

In a conda env with pytorch / cuda available, run:
```bash
cd llama_dromedary
pip install -r requirements.txt
pip install -e .
```

Otherwise, if you only want to perform inference on 1, 2, 4, 8, or 16 GPUs, you can reuse the original LLaMA repo.

```bash
git clone https://github.com/facebookresearch/llama.git
cd llama
pip install -r requirements.txt
pip install -e .
```

## Model Weights

We release Dromedary weights as delta weights to comply with the LLaMA model license. You can add our delta to the original LLaMA weights to obtain the Dromedary weights. Instructions:

1. Get the original LLaMA weights in the huggingface format by following the instructions [here](https://huggingface.co/docs/transformers/main/model_doc/llama).
2. Follow our [inference guide](inference) to see how to deploy Dromedary/LLaMA on your own machine with [model parallel](https://github.com/facebookresearch/fairscale/tree/main/fairscale/nn/model_parallel) (which should be significantly faster than Huggingface's default pipeline parallel when using multiple GPUs).

## Inference

We provide a [chatbot demo](inference/README.md) for Dromedary.

## Training

We provide the full [training pipeline](training/README.md) of Dromedary for reproduction.

## Prompts

All the human annotations used in this project can be found [here](prompts).
