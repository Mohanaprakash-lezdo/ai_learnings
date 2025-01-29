from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer, util
from langchain.text_splitter import RecursiveCharacterTextSplitter
import faiss
import numpy as np
from neo4j import GraphDatabase

# Neo4j Connection Details
NEO4J_URI = "bolt://18.213.110.87"  # Update this if Neo4j is hosted remotely
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "increments-floats-tours"

# Step 1: PDF Text Extraction
def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    reader = PdfReader(pdf_path)
    texts = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            texts.append({"page": i + 1, "text": text})
    return texts

# Step 2: Text Chunking with Metadata
def split_text_with_metadata(texts, chunk_size=500, chunk_overlap=50):
    """Split PDF text into chunks with metadata like page numbers."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = []
    metadata = []

    for entry in texts:
        split_chunks = text_splitter.split_text(entry["text"])
        for chunk in split_chunks:
            chunks.append(chunk)
            metadata.append({"page": entry["page"]})
    return chunks, metadata

# Step 3: Generate Embeddings Using Hugging Face
def generate_embeddings(chunks, model_name="BAAI/bge-base-en"):
    """Generate embeddings for text chunks using HuggingFace models."""
    model = SentenceTransformer(model_name)
    embeddings = model.encode(chunks, show_progress_bar=True, convert_to_tensor=True)
    return embeddings, model

# Step 4: Index Embeddings with FAISS
def index_with_faiss(embeddings):
    """Index embeddings using FAISS."""
    embeddings_np = embeddings.cpu().numpy()  # Convert to numpy
    dimension = embeddings_np.shape[1]
    faiss_index = faiss.IndexFlatIP(dimension)  # Use Inner Product for cosine similarity
    faiss_index.add(embeddings_np)
    print("Embeddings indexed with FAISS.")
    return faiss_index

# Step 5: Store Text Chunks and Metadata in Neo4j
def store_data_in_neo4j(chunks, metadata, embeddings):
    """Store text chunks, metadata, and embeddings in Neo4j."""
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    with driver.session() as session:
        for i, chunk in enumerate(chunks):
            session.run(
                """
                MERGE (c:Chunk {id: $id, text: $text, page: $page, embedding: $embedding})
                """,
                id=i,
                text=chunk,
                page=metadata[i]["page"],
                embedding=embeddings[i].tolist()
            )
    print("Data stored in Neo4j successfully!")
    driver.close()

# Step 6: Query FAISS and Neo4j for Results
def query_system(query, model, faiss_index, chunks, metadata):
    """Query FAISS for similar results and retrieve enriched data from Neo4j."""
    # Generate query embedding
    query_embedding = model.encode(query, convert_to_tensor=True).cpu().numpy()

    # Search FAISS for top matches
    distances, indices = faiss_index.search(np.array([query_embedding]), k=3)

    # Connect to Neo4j
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    print("\nTop Results:")
    for i, idx in enumerate(indices[0]):
        if idx != -1:
            print(f"Rank {i+1} | Page: {metadata[idx]['page']}")
            print(f"Text: {chunks[idx]}")
            print(f"FAISS Score: {distances[0][i]:.4f}")

            # Query Neo4j for related chunks
            with driver.session() as session:
                result = session.run(
                    """
                    MATCH (c:Chunk {id: $id}) RETURN c.page AS page, c.text AS text
                    """,
                    id=idx
                )
                for record in result:
                    print(f"Neo4j Verified Text (Page {record['page']}): {record['text']}")
            print("-" * 50)
    driver.close()

# Main Function
def main():
    pdf_path = "/content/Morgan_ Tujuanza - OCR.pdf"  # Update with your PDF path

    # Step 1: Extract PDF text
    pdf_texts = extract_text_from_pdf(pdf_path)
    print("PDF text extracted successfully!")

    # Step 2: Split text into chunks with metadata
    text_chunks, metadata = split_text_with_metadata(pdf_texts)
    print(f"Text split into {len(text_chunks)} chunks.")

    # Step 3: Generate embeddings
    embeddings, model = generate_embeddings(text_chunks)
    print("Embeddings generated successfully!")

    # Step 4: Index embeddings using FAISS
    faiss_index = index_with_faiss(embeddings)

    # Step 5: Store data in Neo4j
    store_data_in_neo4j(text_chunks, metadata, embeddings)

    # Step 6: Query system
    query = input("Enter your search query: ")
    query_system(query, model, faiss_index, text_chunks, metadata)

if __name__ == "__main__":
    main()
