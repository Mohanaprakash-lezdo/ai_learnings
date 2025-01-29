


import streamlit as st
import warnings
warnings.filterwarnings('ignore')
import os
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import T5ForConditionalGeneration, T5Tokenizer
from PyPDF2 import PdfReader

# Initialize models
@st.cache_resource
def load_models():
    st_model = SentenceTransformer("abhinand/MedEmbed-base-v0.1")
    t5_model = T5ForConditionalGeneration.from_pretrained("t5-small")
    t5_tokenizer = T5Tokenizer.from_pretrained("t5-small")
    return st_model, t5_model, t5_tokenizer

model, summarization_model, summarization_tokenizer = load_models()

# Define Mytryoshka Loss
class MytryoshkaLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(MytryoshkaLoss, self).__init__()
        self.margin = margin

    def forward(self, query_embeddings, target_embeddings):
        distances = torch.norm(query_embeddings - target_embeddings, dim=1)
        return torch.mean(torch.clamp(self.margin - distances, min=0))

# Load data from file
def load_data(file):
    ext = os.path.splitext(file.name)[-1].lower()
    if ext == ".csv":
        df = pd.read_csv(file)
        selected_column = st.selectbox("Select a column for text extraction:", df.columns)
        return df[selected_column].dropna().tolist()
    elif ext == ".pdf":
        reader = PdfReader(file)
        text = ''.join(page.extract_text() for page in reader.pages)
        return text.split("\n")
    elif ext == ".txt":
        return file.read().decode("utf-8").splitlines()
    else:
        st.error("Unsupported file format. Only CSV, PDF, and TXT are supported.")
        return None

# Function to summarize results
def summarize_results(results):
    if not results:
        return "No results to summarize."
    text_to_summarize = " ".join([sentence[0] for sentence in results])
    input_ids = summarization_tokenizer.encode("summarize: " + text_to_summarize, return_tensors="pt", max_length=512, truncation=True)
    summary_ids = summarization_model.generate(input_ids, max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
    return summarization_tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# Retrieve relevant sentences
def derive_relevant_sentences(query, dataset, top_k, similarity_threshold):
    query_embedding = model.encode([query], convert_to_tensor=True)
    dataset_embeddings = model.encode(dataset, convert_to_tensor=True)
    similarities = cosine_similarity(query_embedding.cpu(), dataset_embeddings.cpu())[0]

    sorted_sentences = [
        (dataset[idx], similarities[idx])
        for idx, score in enumerate(similarities)
        if score >= similarity_threshold
    ]
    sorted_sentences = sorted(sorted_sentences, key=lambda x: x[1], reverse=True)
    
    # Return an empty list if no relevant sentences are found
    if not sorted_sentences:
        st.error("The query is out of context! No relevant information found in the dataset.")
        return []
    
    return sorted_sentences[:top_k]

# Evaluate retrieval performance
def evaluate_retrieval(query, relevant_sentences, k):
    retrieved = [sent[0] for sent in relevant_sentences[:k]]
    relevant = [sent[0] for sent in relevant_sentences]

    def mrr_at_k(retrieved, relevant):
        for rank, doc in enumerate(retrieved, start=1):
            if doc in relevant:
                return 1 / rank
        return 0

    def recall_at_k(retrieved, relevant, k=5):
        retrieved_at_k = set(retrieved[:k])
        relevant_set = set(relevant)
        return len(retrieved_at_k.intersection(relevant_set)) / len(relevant_set) if len(relevant_set) > 0 else 1.0

    def ndcg_at_k(retrieved, relevant, k=5):
        dcg = sum(1 / np.log2(i + 2) for i, doc in enumerate(retrieved[:k]) if doc in relevant)
        idcg = sum(1 / np.log2(i + 2) for i in range(min(len(relevant), k)))
        return dcg / idcg if idcg > 0 else 0

    mrr = mrr_at_k(retrieved, relevant)
    recall = recall_at_k(retrieved, relevant, k)
    ndcg = ndcg_at_k(retrieved, relevant, k)
    return mrr, recall, ndcg

# Plot similarity heatmap
def plot_similarity_heatmap(relevant_sentences, query):
    labels = [f"Sentence {i+1}" for i in range(len(relevant_sentences))]
    similarities = [score for _, score in relevant_sentences]
    cmap = sns.color_palette("coolwarm", len(similarities))
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        np.array([similarities]).T,
        annot=True,
        cmap=cmap,
        cbar=True,
        xticklabels=[query],
        yticklabels=labels,
        ax=ax
    )
    plt.title("Similarity Heatmap")
    plt.xlabel("Query")
    plt.ylabel("Top Sentences")
    st.pyplot(fig)

# Fine-tune the embeddings and calculate Mytryoshka loss
def fine_tune_and_calculate_loss(query, relevant_sentences, threshold=1e-5):
    formatted_sentences = [sentence for sentence, _ in relevant_sentences]
    try:
        query_embeddings = model.encode([query], convert_to_tensor=True)
        target_embeddings = model.encode(formatted_sentences, convert_to_tensor=True)

        query_embeddings.requires_grad_()
        target_embeddings.requires_grad_()

        optimizer = torch.optim.Adam([query_embeddings, target_embeddings], lr=1e-5)
        loss_fn = MytryoshkaLoss()

        epoch = 0
        while True:
            epoch += 1
            optimizer.zero_grad()
            loss = loss_fn(query_embeddings, target_embeddings)
            loss.backward()
            optimizer.step()

            # Stop if loss is smaller than the threshold
            if loss.item() <= threshold:
                break

        return loss.item()

    except RuntimeError as e:
        st.error(f"RuntimeError encountered during fine-tuning: {str(e)}")
        return None

# Streamlit App
st.image("https://jobs.lezdotechmed.com/wp-content/uploads/2022/08/lezdo-logo-aw.svg", width=180)  # Adjust the path and size as needed

st.title("MedInfo Retriver")

# Sidebar for Evaluation Metrics
st.sidebar.header("Evaluation Metrics")
top_k = st.sidebar.slider("Number of top relevant sentences to retrieve:", 1, 20, 5)

# Main body for file upload and query input
st.sidebar.markdown("---")


file = st.file_uploader("Upload your dataset (CSV, PDF, or TXT):", type=["csv", "pdf", "txt"])
query = st.text_input("Enter your query:")

# Predefined similarity threshold (hidden from UI)
similarity_threshold = 0.5

if file and query:
    dataset = load_data(file)
    if dataset:
        relevant_sentences = derive_relevant_sentences(query, dataset, top_k, similarity_threshold)
        if relevant_sentences:
            # Calculate Mytryoshka Loss
            mytryoshka_loss = fine_tune_and_calculate_loss(query, relevant_sentences)
            

            # Evaluate retrieval performance
            mrr, recall, ndcg = evaluate_retrieval(query, relevant_sentences, top_k)
            st.sidebar.write(f"**MRR:** {mrr:.4f}")
            st.sidebar.write(f"**Recall:** {recall:.4f}")
            st.sidebar.write(f"**NDCG:** {ndcg:.4f}")
            if mytryoshka_loss is not None:
                st.sidebar.write(f"**Mytryoshka Loss:** {mytryoshka_loss:.4f}")
            # Display Abstractive Summary first
            st.write("### Abstractive Summary")
            st.write(summarize_results(relevant_sentences))

            # Display Top Relevant Sentences next
            st.write("### Top Relevant Sentences")
            for idx, (sentence, score) in enumerate(relevant_sentences):
                st.write(f"{idx + 1}: {sentence} ")

            st.write("### Similarity Heatmap")
            plot_similarity_heatmap(relevant_sentences, query)
            
            
st.markdown("""
    <style>
        .footer {
            position: fixed;
            bottom: 0;
            left: 0;
            width: 100%;
            background-color: #E8ECF5;
            text-align: center;
            padding: 10px;
            font-size: 14px;
            color: black;
            box-shadow: 0px -1px 5px rgba(0,0,0,0.1);
        }
    </style>
""", unsafe_allow_html=True)

# Display footer content
st.markdown("""
    <div class="footer">
        <p>&copy; 2024 MedInfo Retriever. All rights reserved.</p>
    </div>
""", unsafe_allow_html=True)


