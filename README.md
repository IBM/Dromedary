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
cd ..
```

Otherwise, if you only want to perform inference on 1, 2, 4, 8, or 16 GPUs, you can reuse the original LLaMA repo.

```bash
git clone https://github.com/facebookresearch/llama.git
cd llama
pip install -r requirements.txt
pip install -e .
cd ..
```

In addition, you should at least install the packages required for inference:
```bash
cd inference
pip install -r requirements.txt
```

## Model Weights

We release Dromedary weights as delta weights to comply with the LLaMA model license. You can add our delta to the original LLaMA weights to obtain the Dromedary weights. Instructions:

1. Get the original LLaMA weights in the huggingface format by following the instructions [here](https://huggingface.co/docs/transformers/main/model_doc/llama).
2. Follow our [inference guide](inference) to see how to deploy Dromedary/LLaMA on your own machine with [model parallel](https://github.com/facebookresearch/fairscale/tree/main/fairscale/nn/model_parallel) (which should be significantly faster than Huggingface's default pipeline parallel when using multiple GPUs).

## Inference

We provide a [chatbot demo](inference) for Dromedary.

## Training

We provide the full [training pipeline](training) of Dromedary for reproduction.

## Prompts

All the human annotations used in this project can be found [here](prompts).

### TODOs

- [ ] Add the requirements.txt for the training pipeline.
- [ ] Add the evaluation code for TruthfulQA and HHH Eval.
- [ ] Release Dromedary delta weights at Huggingface model hub.
- [ ] Add support for streaming inference in chatbot demo.
- [ ] Fix the Huggingface datasets/accelerate bug of fine-tuning in distributed setting.

### Citation

Please cite the following paper if you use the data or code in this repo.

TBD

### Acknowledgements

We thank Yizhong Wang for providing the code for the parse analysis plot.
We also thank [Meta LLaMA team](https://github.com/facebookresearch/llama), [Standford Alpaca team](https://github.com/tatsu-lab/stanford_alpaca), [Vicuna team](https://github.com/lm-sys/FastChat), [Alpaca-LoRA](https://github.com/tloen/alpaca-lora), and [Huggingface PEFT](https://github.com/huggingface/peft) for their open-source efforts in democratizing large language models.
