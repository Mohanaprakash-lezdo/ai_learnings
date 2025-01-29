import fitz  # PyMuPDF for PDF parsing
from pdf2image import convert_from_path
from PIL import Image
import torch
from transformers import FlamingoTokenizer, FlamingoForConditionalGeneration
import io

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    text = ""
    doc = fitz.open(pdf_path)
    for page in doc:
        text += page.get_text()  # Extract text from each page
    return text

# Function to extract images from PDF
def extract_images_from_pdf(pdf_path):
    images = []
    # Convert PDF pages to images using pdf2image
    pages = convert_from_path(pdf_path, 300)  # 300 dpi is high quality
    for page in pages:
        # Convert each page to an image object
        images.append(page)
    return images

# Function to process PDF with Flamingo Mini (or any multimodal vision-language model)
def process_pdf_with_flamingo(pdf_path, question):
    # 1. Extract text and images from PDF
    text = extract_text_from_pdf(pdf_path)
    images = extract_images_from_pdf(pdf_path)

    # 2. Initialize Flamingo Mini (Placeholder here: model should be fine-tuned or loaded correctly)
    model_name = "flamingo-mini"  # Replace with the actual Flamingo Mini model or another model
    tokenizer = FlamingoTokenizer.from_pretrained(model_name)
    model = FlamingoForConditionalGeneration.from_pretrained(model_name)

    # 3. Tokenize the question and PDF text input for Flamingo
    combined_input = question + "\n\n" + text  # Combine the question with the extracted text
    inputs = tokenizer(combined_input, return_tensors="pt", truncation=True, padding=True)

    # 4. If you want to use the images, convert them into tensor format (e.g., PIL -> Tensor)
    image_tensors = []
    for img in images:
        img_tensor = torch.tensor(img.convert("RGB"))
        image_tensors.append(img_tensor)

    # 5. Model inference (text + image)
    # The model input format might be different based on your exact task
    try:
        outputs = model.generate(input_ids=inputs["input_ids"], pixel_values=image_tensors)
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return answer
    except Exception as e:
        return f"Error in generating output: {str(e)}"

# Example usage:
# Path to the PDF
pdf_path = '/content/Redacted.pdf'

# User's question about the PDF
user_question = input("")

# Get the model's answer
answer = process_pdf_with_flamingo(pdf_path, user_question)
print(f"Answer: {answer}")
