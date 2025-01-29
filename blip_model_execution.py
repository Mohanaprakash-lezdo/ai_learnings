import os
from pdf2image import convert_from_path
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

# Step 1: Convert PDF pages to images
def convert_pdf_to_images(pdf_path, output_folder="/tmp/pdf_images"):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Convert PDF to images (one per page)
    images = convert_from_path(pdf_path)
    image_paths = []
    for i, img in enumerate(images):
        img_path = os.path.join(output_folder, f"page_{i + 1}.jpg")
        img.save(img_path, 'JPEG')
        image_paths.append(img_path)
    
    return image_paths

# Step 2: Initialize a pre-trained VQA model (BLIP)
def initialize_vqa_model():
    # Load pre-trained BLIP model and processor for VQA
    processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-vqa-base")
    
    # Move model to CPU
    model.to('cpu')
    return processor, model

# Step 3: Answer questions based on the image using VQA (no embeddings)
def answer_question_from_image(image_path, processor, model, question):
    # Open the image from the file path
    raw_image = Image.open(image_path)

    # Preprocess the image and the question for VQA
    inputs = processor(images=raw_image, text=question, return_tensors="pt")

    # Move inputs to CPU
    inputs = {key: value.to('cpu') for key, value in inputs.items()}

    # Get the answer from the model
    outputs = model.generate(**inputs)
    
    # Decode the answer and return it
    answer = processor.decode(outputs[0], skip_special_tokens=True)
    return answer

# Step 4: Full pipeline to process PDF, convert to images, and answer questions
def vlm_pipeline(pdf_path, question):
    # Step 1: Convert PDF to images
    image_paths = convert_pdf_to_images(pdf_path)
    
    # Step 2: Initialize the VQA model (BLIP)
    processor, model = initialize_vqa_model()
    
    # Step 3: Process each image and answer the question
    answers = {}
    for image_path in image_paths:
        answer = answer_question_from_image(image_path, processor, model, question)
        answers[image_path] = answer
    
    return answers

# Example usage
pdf_path = "/content/Redacted.pdf"  # Specify the path to your PDF file
question = "What is the history of present illness in the first image?"  # Example question

# Running the VLM pipeline
answers = vlm_pipeline(pdf_path, question)

# Output answers for each image in the PDF
for image_path, answer in answers.items():
    print(f"Answer for {image_path}: {answer}")
