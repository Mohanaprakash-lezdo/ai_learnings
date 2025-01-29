import spacy
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics.pairwise import cosine_similarity

# --- Mytryoshka Loss ---
class MytryoshkaLoss(nn.Module):
    def __init__(self):
        super(MytryoshkaLoss, self).__init__()

    def forward(self, anchor, positive, negative):
        # Compute Euclidean distances
        positive_distance = torch.norm(anchor - positive, p=2, dim=1)
        negative_distance = torch.norm(anchor - negative, p=2, dim=1)
        # Loss: Maximize positive similarity, minimize negative
        loss = torch.clamp(positive_distance - negative_distance + 1, min=0).mean()
        return loss

# --- Step 1: Process Unstructured Text ---
def process_unstructured_text(text):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents]

# --- Step 2: Generate Questions ---
def generate_questions(sentences):
    question_generator = pipeline("text2text-generation", model="valhalla/t5-small-qg-prepend")
    qa_pairs = []
    for sentence in sentences:
        question = question_generator(f"generate question: {sentence}")[0]["generated_text"]
        qa_pairs.append({"query": question, "response": sentence})
    return qa_pairs

# --- Step 3: Query-Response System with Fine-Tuning ---
class QueryResponseSystem:
    def __init__(self, qa_pairs):
        self.qa_pairs = qa_pairs
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.loss_fn = MytryoshkaLoss()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=2e-5)
        self.response_embeddings = None

    def fine_tune(self, epochs=1, batch_size=1):
        # Prepare data
        queries = [pair["query"] for pair in self.qa_pairs]
        responses = [pair["response"] for pair in self.qa_pairs]

        # The embeddings should be recalculated inside the loop for each batch.
        # This ensures they are part of the computation graph and their gradients are tracked.
        for epoch in range(epochs):
            total_loss = 0
            for i in range(0, len(queries), batch_size):
                batch_queries = queries[i:i + batch_size]
                batch_responses = responses[i:i + batch_size]

                # Encode embeddings within the training loop with `requires_grad=True`
                query_embeddings = self.model.encode(batch_queries, convert_to_tensor=True)
                query_embeddings = query_embeddings.detach().requires_grad_(True)  # Ensure gradients are tracked

                response_embeddings = self.model.encode(batch_responses, convert_to_tensor=True)
                response_embeddings = response_embeddings.detach().requires_grad_(True)  # Ensure gradients are tracked

                # Create negative samples
                negatives = response_embeddings + torch.randn_like(response_embeddings) * 0.5

                self.optimizer.zero_grad()
                loss = self.loss_fn(query_embeddings, response_embeddings, negatives)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")

        # Update response embeddings after fine-tuning
        self.response_embeddings = self.model.encode(responses, convert_to_tensor=True)

    def get_response(self, user_query):
        user_embedding = self.model.encode(user_query, convert_to_tensor=True)
        similarities = util.cos_sim(user_embedding, self.response_embeddings)[0]
        best_match_idx = similarities.argmax().item()
        best_similarity = similarities[best_match_idx].item()
        if best_similarity > 0.5:
            return self.qa_pairs[best_match_idx]["response"]
        else:
            return "No relevant response found."

# --- Metrics ---
def evaluate_metrics(retrieved, relevant,k=5):
    def mrr_at_k(retrieved, relevant):
        for rank, doc in enumerate(retrieved[:k], start=1):
            if doc in relevant:
                return 1 / rank
        return 0

    def recall_at_k(retrieved, relevant, k):
        relevant_set = set(relevant)
        retrieved_set = set(retrieved[:k])
        return len(relevant_set & retrieved_set) / len(relevant_set)

    def ndcg_at_k(retrieved, relevant, k):
        dcg = sum(1 / np.log2(i + 2) for i, doc in enumerate(retrieved[:k]) if doc in relevant)
        idcg = sum(1 / np.log2(i + 2) for i in range(min(len(relevant), k)))
        return dcg / idcg if idcg > 0 else 0

    mrr = mrr_at_k(retrieved, relevant)
    recall = recall_at_k(retrieved, relevant, k)
    ndcg = ndcg_at_k(retrieved, relevant, k)
    return {"MRR": mrr, "Recall@K": recall, "NDCG@K": ndcg}

# --- Main Script ---
if __name__ == "__main__":
    # Unstructured Text
    unstructured_text = """
    Medical records are critical for patient care. They help doctors understand a patient's history.
    Digitizing records improves efficiency and reduces errors.
    Treatment plans are essential for minimizing complications.
    """

    # Step 1: Process Text
    sentences = process_unstructured_text(unstructured_text)

    # Step 2: Generate QA Pairs
    qa_pairs = generate_questions(sentences)
    print("Generated QA Pairs:")
    print(pd.DataFrame(qa_pairs))

    # Step 3: Initialize Query-Response System and Fine-Tune
    query_response_system = QueryResponseSystem(qa_pairs)
    query_response_system.fine_tune(epochs=1)

    # User Query and Response
    user_query = "Why are medical records important?"
    response = query_response_system.get_response(user_query)
    print(f"\nUser Query: {user_query}")
    print(f"Response: {response}")

    # Step 4: Metrics Evaluation
    relevant_responses = [qa["response"] for qa in qa_pairs]
    retrieved_responses = [
        qa_pairs[i]["response"]
        for i in util.cos_sim(query_response_system.model.encode(user_query), query_response_system.response_embeddings)[0]
        .argsort(descending=True)[:5]
        .tolist()
    ]
    metrics = evaluate_metrics(retrieved_responses, relevant_responses, k=5)
    print("\nEvaluation Metrics:")
    print(f"MRR: {metrics['MRR']:.4f}")
    print(f"Recall@5: {metrics['Recall@K']:.4f}")
    print(f"NDCG@5: {metrics['NDCG@K']:.4f}")
