# Fine-Tuning Mistral-7B with LoRA for Sentiment Analysis

This project demonstrates parameter-efficient fine-tuning of Mistral-7B using LoRA (Low-Rank Adaptation) and 4-bit quantization for sentiment analysis on the IMDB dataset.

## Overview

The goal is to fine-tune a large language model to classify movie reviews as positive or negative, using minimal computational resources through:

- **4-bit quantization** (QLoRA) to reduce memory footprint
- **LoRA adapters** to train only ~1.1% of the model parameters
- **Supervised Fine-Tuning (SFT)** with instruction-style prompts

## Requirements

```bash
pip install transformers datasets peft bitsandbytes accelerate evaluate trl
```

**Hardware**: A GPU with at least 16GB VRAM is recommended. The notebook was tested on Google Colab with a T4 GPU.

## Project Structure

```
.
├── FineTuningWithLorA.ipynb    # Main notebook
└── README.md
```

## Method

### Quantization

The model is loaded in 4-bit precision using BitsAndBytes:

```python
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)
```

### LoRA Configuration

LoRA adapters are applied to all attention and MLP projection layers:

| Parameter | Value |
|-----------|-------|
| Rank (r) | 16 |
| Alpha | 32 |
| Dropout | 0.05 |
| Target modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |

This results in training approximately 0.5% of the total parameters.

### Dataset Preparation

The IMDB dataset is formatted as instruction-response pairs:

```
### Instruction:
Analyze the sentiment of the following movie review and respond with only one word: positive or negative.

### Review:
{review_text}

### Sentiment:
{positive|negative}
```

**Label masking**: The loss is computed only on the response tokens (`positive`/`negative` and EOS). Prompt tokens and padding are masked with -100.

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Epochs | 3 |
| Batch size | 4 |
| Gradient accumulation | 4 |
| Learning rate | 2e-4 |
| Optimizer | paged_adamw_8bit |
| Warmup steps | 50 |

## Usage

1. Open the notebook in Google Colab or a local Jupyter environment
2. Run all cells sequentially
3. The fine-tuned adapter will be saved to `./mistral-sentiment-ita`

## Results

Training is performed on 2,000 examples with evaluation on 400 examples. Checkpoints are saved every 100 steps.

## References

- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)
- [Mistral-7B](https://huggingface.co/mistralai/Mistral-7B-v0.1)
- [PEFT Library](https://github.com/huggingface/peft)

## License

This project is for educational purposes.
