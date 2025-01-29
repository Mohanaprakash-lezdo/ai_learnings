pip install pdf2image Pillow requests transformers torch
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
from pdf2image import convert_from_path
from PIL import Image
import requests
import torch
import pytesseract

# Load the processor and model
processor = AutoProcessor.from_pretrained(
    'allenai/Molmo-7B-D-0924',
    trust_remote_code=True,
    torch_dtype='auto',
    device_map='auto'
)

model = AutoModelForCausalLM.from_pretrained(
    'allenai/Molmo-7B-D-0924',
    trust_remote_code=True,
    torch_dtype='auto',
    device_map='auto'
)

# Function to convert PDF to images
def pdf_to_images(pdf_path):
    images = convert_from_path(pdf_path, dpi=300)
    return images

# Function to extract text from images
def extract_text_from_image(image):
    return pytesseract.image_to_string(image)

# Function to process a question and generate an answer
def answer_question(question, document_text):
    # Prepare the text input for the model
    inputs = processor.process(
        text=f"Question: {question}\nContext: {document_text}"
    )
    # Generate an answer
    output = model.generate(
        inputs['input_ids'],
        max_new_tokens=200,
        pad_token_id=processor.tokenizer.pad_token_id
    )
    answer = processor.tokenizer.decode(output[0], skip_special_tokens=True)
    return answer

# Load and process PDF
pdf_path = "/content/Redacted.pdf"  # Replace with your PDF file path
images = pdf_to_images(pdf_path)

# Extract text from the PDF
document_text = ""
for image in images:
    document_text += extract_text_from_image(image) + "\n"
document=document_text
# Wait for a user query
print("PDF processing complete. Ask your question about the document:")
while True:
    user_question = input("Your question: ")
    if user_question.lower() in ["exit", "quit"]:
        break
    response = answer_question(user_question, document)
    print(f"Answer: {response}")

import torch

def answer_question(question, document_text):
    # Prepare the text input for the model
    inputs = processor.process(
        text=f"Question: {question}\nContext: {document_text}",
        return_tensors="pt"  # Ensure PyTorch tensors are returned
    )
    
    # Check input tensor dimensions
    if 'input_ids' not in inputs:
        raise ValueError("Error: 'input_ids' not found in processed inputs.")
    
    input_ids = inputs['input_ids']
    attention_mask = inputs.get('attention_mask', torch.ones_like(input_ids))  # Default to ones if missing
    
    print(f"Input IDs shape: {input_ids.shape}")  # Debug: Check shape
    print(f"Attention Mask shape: {attention_mask.shape}")  # Debug: Check shape
    
    # Ensure tensors are moved to the same device as the model
    input_ids = input_ids.to(model.device)
    attention_mask = attention_mask.to(model.device)

    # Generate an answer
    output = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=200,
        pad_token_id=processor.tokenizer.pad_token_id
    )
    
    # Decode the generated output
    answer = processor.tokenizer.decode(output[0], skip_special_tokens=True)
    return answer

# Loop to accept user questions
while True:
    user_question = input("Your question: ")
    if user_question.lower() in ["exit", "quit"]:
        break
    # Check if document_text is valid
    if document_text:  
        try:
            response = answer_question(user_question, document_text)
            print(f"Answer: {response}")
        except Exception as e:
            print(f"Error: {e}")
    else:
        print("Error: Document text is empty. Please provide a valid document.")

