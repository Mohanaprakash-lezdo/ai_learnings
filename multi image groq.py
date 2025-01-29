from groq import Groq
import base64
from PIL import Image

# Function to encode an image into base64
def encode_image(image_path):
    """Reads an image from the given path and returns its base64 encoded string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Function to combine images into one (side-by-side)
def combine_images(image_paths):
    """Combines multiple images into a single image side by side."""
    images = [Image.open(image_path) for image_path in image_paths]
    
    # Assume all images are the same size for simplicity (adjust as needed)
    total_width = sum(image.width for image in images)
    max_height = max(image.height for image in images)

    # Create a new blank image with enough width to hold all images
    combined_image = Image.new("RGB", (total_width, max_height))

    # Paste all images into the combined image
    x_offset = 0
    for image in images:
        combined_image.paste(image, (x_offset, 0))
        x_offset += image.width

    # Save the combined image
    combined_image_path = "combined_image.jpg"
    combined_image.save(combined_image_path)
    
    return combined_image_path

# List of image paths (replace these with your actual image paths)
image_paths = ["/content/2fd2a4824163d61e1197f285adfdd1d3.jpg", "/content/94474673.webp"]  # Add the actual paths to your images

# Combine the images into one composite image
combined_image_path = combine_images(image_paths)

# Encode the combined image into base64
encoded_image = encode_image(combined_image_path)

# Function to send the combined image to the Groq model
def process_image_with_groq(encoded_image, question):
    # Create image URL object from the encoded image
    image_object = {
        "type": "image_url",
        "image_url": {
            "url": f"data:image/jpeg;base64,{encoded_image}"
        }
    }

    # Initialize the Groq client with your API key
    client = Groq(api_key="")  # Replace with your API key

    # Make the request with the question and the combined image
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_question},  # User's dynamic question
                ] + [image_object],  # Add the combined image to the content
            }
        ],
        model="llama-3.2-11b-vision-preview",  # Vision model you're using for image processing
    )

    # Print the response from the model
    print(chat_completion.choices[0].message.content)

# Example usage: Ask a question about the combined image
user_question = input()

# Process the image and get the model's response
process_image_with_groq(encoded_image, user_question)
