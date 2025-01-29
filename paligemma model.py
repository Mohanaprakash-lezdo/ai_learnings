# Install required packages
!pip install transformers huggingface_hub

from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
from huggingface_hub import login
from PIL import Image
import torch
import base64
from io import BytesIO
import os

# Define the model ID
model_id = "google/paligemma-3b-mix-224"

# ----- Hugging Face Login -----
# Log in to your Hugging Face account
login(token='')  
# ------------------------------

# Function to convert image to base64 format
def image_to_base64(image_path):
    try:
        # Open the image file
        with Image.open(image_path) as img:
            # Convert image to byte stream
            buffered = BytesIO()
            img.save(buffered, format="PNG")  # Save as PNG or any other format
            img_byte_arr = buffered.getvalue()

            # Convert to base64
            img_base64 = base64.b64encode(img_byte_arr).decode("utf-8")
            return img_base64
    except Exception as e:
        raise ValueError(f"Failed to convert image to base64. Error: {e}")

# Specify the local image path
image_path = "/content/Screenshot (3).png"  # Replace with your image path

# Check if the image exists
if not os.path.exists(image_path):
    raise FileNotFoundError(f"Image file not found at path: {image_path}")

# Load the image as a PIL Image object
image = Image.open(image_path).convert("RGB")  # Load and ensure RGB format


# Load the model and processor
model = PaliGemmaForConditionalGeneration.from_pretrained(model_id).eval()
processor = AutoProcessor.from_pretrained(model_id)

# Define the prompt
prompt = "explain the evaluation notes on 23/10/2022"  # Instruction for the model to caption

# Process the inputs (using the PIL Image object)
model_inputs = processor(text=prompt, images=image, return_tensors="pt")  # Pass the PIL Image
input_len = model_inputs["input_ids"].shape[-1]

# Generate the caption
with torch.inference_mode():
    generation = model.generate(**model_inputs, max_new_tokens=100, do_sample=False)
    generation = generation[0][input_len:]
    decoded = processor.decode(generation, skip_special_tokens=True)
    print(decoded)