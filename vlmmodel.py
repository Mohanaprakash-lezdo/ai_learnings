from groq import Groq
import base64

# Function to encode the image into base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Path to your image (change this to your actual image path)
image_path = "/content/2fd2a4824163d61e1197f285adfdd1d3.jpg"

# Getting the base64 string of the image
base64_image = encode_image(image_path)

# Get the user question dynamically (e.g., from a form, API, or command-line input)
user_question = input(" ")  # Dynamic user question

# Initialize the Groq client with your API key
client = Groq(api_key="")

# Making the request with the dynamic question and image
chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_question},  # User's dynamic question
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}",  # Base64 image included
                    },
                },
            ],
        }
    ],
    model="llama-3.2-90b-vision-preview",  # Model you're using for image processing
)

# Print the response
print(chat_completion.choices[0].message.content)

