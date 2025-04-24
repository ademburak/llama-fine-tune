# Llama 3.2 Vision Fine-tuning with Unsloth

This project implements fine-tuning of Llama 3.2 Vision model using Unsloth's optimized training approach.

## Features

- Uses Unsloth's optimized training (2x faster, 70% less VRAM)
- 4-bit quantization for memory efficiency
- LoRA fine-tuning with optimized settings
- Supports long context (up to 2048 tokens)

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Prepare your dataset:
   - Your dataset should be in a format compatible with HuggingFace datasets
   - Each example should have an 'image' and 'caption' field
   - Modify the `prepare_dataset()` function in `train.py` to load your specific dataset

## Training

To start training, run:
```bash
python train.py
```

## Model Configuration

The training uses the following default parameters:
- Model: Llama 3.2 Vision 11B
- LoRA rank: 16
- LoRA alpha: 16
- Learning rate: 2e-4
- Batch size: 4
- Gradient accumulation steps: 4
- Max sequence length: 2048
- Number of epochs: 3

## Output

The fine-tuned model will be saved in the `./final_model` directory.

## Notes

- Make sure you have sufficient GPU memory (recommended: 24GB+)
- The training progress will be logged to Weights & Biases
- You can modify the training parameters in the `TrainingArguments` section of `train.py` # llama-fine-tune
# llama-fine-tune
# llama-fine-tune
