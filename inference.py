import os
import torch
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request
from PIL import Image
import io
from unsloth import FastVisionModel
from transformers import AutoTokenizer, AutoImageProcessor, TextStreamer
import uuid
import logging
from safetensors.torch import load_file
import random

# Enable CUDA optimizations
torch.backends.cuda.matmul.allow_tf32 = True  # Allow TF32 on matmul
torch.backends.cudnn.allow_tf32 = True  # Allow TF32 on cudnn
torch.backends.cudnn.benchmark = True  # Enable cudnn auto-tuner
torch.backends.cuda.enable_mem_efficient_sdp = True  # Enable memory-efficient attention
torch.backends.cuda.enable_flash_sdp = True  # Enable Flash Attention if available

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI()

# Create necessary directories
os.makedirs("templates", exist_ok=True)
os.makedirs("static", exist_ok=True)
os.makedirs("uploads", exist_ok=True)

# Initialize templates
templates = Jinja2Templates(directory="templates")

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Global variables for model and tokenizer
model = None
tokenizer = None
image_processor = None

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
        
        # Enable model for inference
        FastVisionModel.for_inference(model)
        model.eval()
        
        # Load the image processor with optimized settings
        image_processor = AutoImageProcessor.from_pretrained(
            "unsloth/Llama-3.2-11B-Vision-Instruct-bnb-4bit",
            do_resize=True,
            do_center_crop=True
        )
        
        # Load the fine-tuned weights if they exist
        adapter_path = "final_model/adapter_model.safetensors"
        if os.path.exists(adapter_path):
            logger.info("Loading fine-tuned weights from safetensors...")
            try:
                state_dict = load_file(adapter_path)
                model.load_state_dict(state_dict, strict=False)
                logger.info("Successfully loaded fine-tuned weights")
            except Exception as e:
                logger.error(f"Error loading safetensors: {str(e)}")
                raise
        else:
            logger.warning("Fine-tuned weights not found, using base model")
        
        # Clean up CUDA memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
        model.eval()
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

# Create HTML template
with open("templates/index.html", "w") as f:
    f.write("""
<!DOCTYPE html>
<html>
<head>
    <title>Vision Language Model Inference</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .image-preview {
            max-width: 400px;
            margin-top: 10px;
            border-radius: 4px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .result {
            margin-top: 20px;
            padding: 15px;
            border: 1px solid #ddd;
            border-radius: 4px;
            background-color: #f9f9f9;
        }
        .loading {
            display: none;
            margin-top: 10px;
            color: #666;
        }
        button {
            background-color: #007bff;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 4px;
            cursor: pointer;
        }
        button:hover {
            background-color: #0056b3;
        }
        input[type="text"] {
            width: 100%;
            padding: 8px;
            border: 1px solid #ddd;
            border-radius: 4px;
            margin-top: 5px;
        }
        .error {
            color: #dc3545;
            margin-top: 10px;
            padding: 10px;
            border: 1px solid #dc3545;
            border-radius: 4px;
            background-color: #f8d7da;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Vision Language Model Inference</h1>
        <form id="uploadForm">
            <div>
                <label for="image">Upload Image:</label><br>
                <input type="file" id="image" name="image" accept="image/*" required>
                <img id="preview" class="image-preview" style="display: none;">
            </div>
            <div style="margin-top: 10px;">
                <label for="prompt">Prompt:</label><br>
                <input type="text" id="prompt" name="prompt" style="width: 100%;" 
                       value="Describe this image in detail" required>
            </div>
            <button type="submit" style="margin-top: 10px;">Generate</button>
        </form>
        <div id="loading" class="loading">Processing...</div>
        <div id="error" class="error" style="display: none;"></div>
        <div id="result" class="result" style="display: none;"></div>
    </div>

    <script>
        // Preview image
        document.getElementById('image').addEventListener('change', function(e) {
            const preview = document.getElementById('preview');
            const file = e.target.files[0];
            if (file) {
                preview.src = URL.createObjectURL(file);
                preview.style.display = 'block';
            }
        });

        // Handle form submission
        document.getElementById('uploadForm').addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const formData = new FormData();
            formData.append('image', document.getElementById('image').files[0]);
            formData.append('prompt', document.getElementById('prompt').value);

            document.getElementById('loading').style.display = 'block';
            document.getElementById('result').style.display = 'none';
            document.getElementById('error').style.display = 'none';

            try {
                const response = await fetch('/generate', {
                    method: 'POST',
                    body: formData
                });
                
                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }
                
                const data = await response.json();
                if (data.error) {
                    throw new Error(data.error);
                }
                
                document.getElementById('result').textContent = data.response;
                document.getElementById('result').style.display = 'block';
            } catch (error) {
                console.error('Error:', error);
                document.getElementById('error').textContent = error.message;
                document.getElementById('error').style.display = 'block';
            } finally {
                document.getElementById('loading').style.display = 'none';
            }
        });
    </script>
</body>
</html>
    """)

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/generate")
async def generate(
    image: UploadFile = File(...),
    prompt: str = Form(...)
):
    try:
        # Check if model is loaded
        if model is None or tokenizer is None or image_processor is None:
            load_model()
        
        # Save uploaded image
        image_filename = f"{uuid.uuid4()}_{image.filename}"
        image_path = os.path.join("uploads", image_filename)
        
        # Read and save the image
        contents = await image.read()
        with open(image_path, "wb") as f:
            f.write(contents)
        
        # Load and process image
        img = Image.open(image_path).convert("RGB")
        
        # Format the instruction
        instruction = prompt
        
        # Create messages in the correct format
        messages = [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": instruction}
            ]}
        ]
        
        # Apply chat template
        input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        
        # Tokenize with the correct format
        inputs = tokenizer(
            img,
            input_text,
            add_special_tokens=False,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
            return_attention_mask=True
        ).to(model.device, non_blocking=True)
        
        # Print token information
        logger.info(f"Input tokens: {len(inputs['input_ids'][0])}")
        
        # Generate with optimized settings
        with torch.inference_mode(), torch.cuda.amp.autocast():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True,
                num_beams=1,
                early_stopping=True
            )
        
        # Print output token information
        logger.info(f"Generated tokens: {len(outputs[0]) - len(inputs['input_ids'][0])}")
        
        # Decode response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Clean up the response by removing the prompt
        response = response.replace(input_text, "").strip()
        
        return {
            "response": response,
            "token_info": {
                "input_tokens": len(inputs['input_ids'][0]),
                "generated_tokens": len(outputs[0]) - len(inputs['input_ids'][0])
            }
        }
        
    except Exception as e:
        logger.error(f"Error during generation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    # Load model before starting the server
    load_model()
    uvicorn.run(app, host="0.0.0.0", port=8000) 