import os
from unsloth import FastLanguageModel, FastVisionModel
import torch
from datasets import Dataset
from transformers import TrainingArguments
from trl import SFTTrainer
import wandb
import pandas as pd
from PIL import Image
from transformers import AutoImageProcessor
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Enable CUDA optimizations
torch.backends.cuda.matmul.allow_tf32 = True  # Allow TF32 on matmul
torch.backends.cudnn.allow_tf32 = True  # Allow TF32 on cudnn
torch.backends.cudnn.benchmark = True  # Enable cudnn auto-tuner
torch.backends.cuda.enable_mem_efficient_sdp = True  # Enable memory-efficient attention
torch.backends.cuda.enable_flash_sdp = True  # Enable Flash Attention if available

print(torch.cuda.mem_get_info())
# Initialize wandb
wandb.init(project="llama-vision-fine-tuning")

# Declare global variables
model = None
tokenizer = None
image_processor = None

# Set up model and tokenizer - using Llama which fits in 16GB
def load_model():
    global model, tokenizer, image_processor
    try:
        logger.info("Loading model and tokenizer...")
        model, tokenizer = FastVisionModel.from_pretrained(
            "unsloth/Llama-3.2-11B-Vision-Instruct-bnb-4bit",
            load_in_4bit=True,
            use_gradient_checkpointing="unsloth",
            device_map="auto"
        )
        
        # Load the image processor with optimized settings
        image_processor = AutoImageProcessor.from_pretrained(
            "unsloth/Llama-3.2-11B-Vision-Instruct-bnb-4bit",
            do_resize=True,
            do_center_crop=True
        )
        
        # Configure LoRA first
        model = FastVisionModel.get_peft_model(
            model,
            r=16,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ],
            lora_alpha=16,
            lora_dropout=0,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=42,
            max_seq_length=2048
        )
        
        # Clean up CUDA memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

# Load and prepare dataset
def prepare_dataset():
    # Load captions from CSV
    image_dir = "data/images"
    caption_file = "data/captions.csv"
    
    # Check if directories and files exist
    if not os.path.exists(image_dir):
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    if not os.path.exists(caption_file):
        raise FileNotFoundError(f"Caption file not found: {caption_file}")
    
    # Read captions from CSV
    try:
        df = pd.read_csv(caption_file)
        if df.empty:
            raise ValueError("Caption file is empty")
        if "image_path" not in df.columns or "caption" not in df.columns:
            raise ValueError("Caption file must contain 'image_path' and 'caption' columns")
    except pd.errors.EmptyDataError:
        raise ValueError("Caption file is empty or not properly formatted")
    
    # Create dataset
    data = []
    for _, row in df.iterrows():
        img_path = os.path.join(image_dir, row["image_path"])
        if os.path.exists(img_path):
            data.append({
                "image_path": img_path,
                "caption": row["caption"]
            })
        else:
            print(f"Warning: Image not found: {img_path}")
    
    if not data:
        raise ValueError("No valid image-caption pairs found. Please check your data/images directory and captions.csv file")
    
    print(f"Found {len(data)} valid image-caption pairs")
    dataset = Dataset.from_list(data)
    
    def format_instruction(example):
        image = Image.open(example["image_path"]).convert("RGB")
        return {
            "text": f"<image>Describe this image in detail: {example['caption']}</image>",
            "image": image
        }
    
    dataset = dataset.map(format_instruction)
    return dataset

# Training arguments - minimal version for 50 epochs
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=50,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    bf16=True,
    optim="adamw_8bit",
)

try:
    # Load the model first
    load_model()
    
    # Initialize trainer
    dataset = prepare_dataset()
    trainer = SFTTrainer(
        model=model,  # Now model is defined
        train_dataset=dataset,
        tokenizer=tokenizer,
        args=training_args,
        max_seq_length=2048,
    )

    # Start training
    trainer.train()

    # Save the model
    trainer.save_model("./final_model")
except Exception as e:
    print(f"Error during training: {str(e)}")
    raise 