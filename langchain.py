import os
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Set Hugging Face API Key (Get yours at https://huggingface.co/settings/tokens)
os.environ["HF_AUTH_TOKEN"] = ""  # Replace with your Hugging Face API Token

# Load a pre-trained Hugging Face conversational model
# Changed model_name to microsoft/DialoGPT-medium
model_name = "microsoft/DialoGPT-medium"  # A free and open-source conversational model from Hugging Face
tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=True)
model = AutoModelForCausalLM.from_pretrained(model_name, use_auth_token=True)

# Create a conversational pipeline using Hugging Face
chat_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Use Langchain to wrap the Hugging Face pipeline into an LLM (Large Language Model)
llm = HuggingFacePipeline(pipeline=chat_pipeline)

# Define the prompt template
prompt_template = PromptTemplate(
    input_variables=["question"],
    template="You are a helpful assistant. Answer the following question:\n\nQuestion: {question}\nAnswer:"
)

# Create the LLM chain
llm_chain = LLMChain(llm=llm, prompt=prompt_template)

# Function to interact with the model
def ask_question(question):
    response = llm_chain.run(question)
    return response

# Main loop for user interaction
if __name__ == "__main__":
    print("AI Chatbot Initialized. Ask your questions! Type 'exit' to quit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break
        try:
            answer = ask_question(user_input)
            print(f"Assistant: {answer.strip()}")
        except Exception as e:
            print(f"An error occurred: {e}")