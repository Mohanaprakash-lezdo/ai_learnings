import os
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer, util
from langchain.text_splitter import RecursiveCharacterTextSplitter
from neo4j import GraphDatabase

def process_pdf(file_path):
    """
    Extract text from a PDF and return it as a single document string.
    """
    print("Processing PDF...")
    try:
        reader = PdfReader(file_path)
        document = "\n".join([page.extract_text() or "" for page in reader.pages])
        return document
    except Exception as e:
        print(f"Error processing PDF: {e}")
        return None

def create_embedding_model(model_name="sentence-transformers/all-MiniLM-L6-v2"):
    """
    Creates a SentenceTransformer model for generating embeddings.
    """
    print(f"Loading embedding model: {model_name}")
    return SentenceTransformer(model_name)

def chunk_text(document, chunk_size=500, chunk_overlap=100):
    """
    Split the document into smaller chunks using LangChain's text splitter.
    """
    print("Splitting text into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    chunks = splitter.split_text(document)
    print(f"Text has been split into {len(chunks)} chunks.")
    return chunks

def connect_to_neo4j(uri, user, password):
    """
    Connects to the Neo4j database and returns a driver object.
    """
    print("Connecting to Neo4j...")
    driver = GraphDatabase.driver(uri, auth=(user, password))
    return driver

def add_chunks_to_neo4j(driver, chunks, embeddings, pdf_name):
    """
    Add chunks and their embeddings to the Neo4j database.
    """
    print("Adding chunks to Neo4j...")
    with driver.session() as session:
        # Create a node for the PDF
        session.run("MERGE (pdf:PDF {name: $name})", name=pdf_name)

        # Add each chunk as a node
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            session.run(
                """
                CREATE (c:Chunk {chunk_id: $chunk_id, content: $content, embedding: $embedding})
                WITH c
                MATCH (pdf:PDF {name: $pdf_name})
                MERGE (pdf)-[:CONTAINS]->(c)
                """,
                chunk_id=f"{pdf_name}_chunk_{i+1}",
                content=chunk,
                embedding=embedding.tolist(),
                pdf_name=pdf_name,
            )

    print("Chunks and relationships successfully added to Neo4j!")

def query_neo4j(driver, query, embedding_model, chunks, embeddings, pdf_name):
    """
    Handles user queries by finding the most similar chunks and adding the query to the Neo4j database.
    """
    print("Processing user query...")
    query_embedding = embedding_model.encode(query, convert_to_tensor=True)

    # Find similarity scores
    scores = util.cos_sim(query_embedding, embeddings)

    # Adjust 'k' based on the number of chunks
    k = min(5, len(chunks))  # Select up to 5 or the total number of chunks
    if k == 0:
        print("No chunks are available for similarity comparison.")
        return []

    top_indices = scores.topk(k=k).indices.squeeze().tolist()  # Ensure indices is a flat list
    if isinstance(top_indices, int):  # Handle case when only one index is returned
        top_indices = [top_indices]

    top_chunks = [(i, chunks[i], scores[0, i].item()) for i in top_indices]

    # Store query and relationships in Neo4j
    with driver.session() as session:
        # Add query as a node
        session.run("CREATE (q:Query {text: $text})", text=query)

        # Link query to the relevant chunks
        for i, chunk, score in top_chunks:
            session.run(
                """
                MATCH (c:Chunk {chunk_id: $chunk_id})
                MERGE (q:Query {text: $text})-[:RELATED {score: $score}]->(c)
                """,
                chunk_id=f"{pdf_name}_chunk_{i+1}",
                text=query,
                score=score,
            )

    return top_chunks

def main():
    # Input PDF file
    file_path = ("/content/8UyYDr4oK9.pdf")
    if not os.path.exists(file_path):
        print("The specified file does not exist.")
        return

    # Process the PDF
    document = process_pdf(file_path)
    if not document:
        print("No text found in the PDF.")
        return

    # Chunk the text
    chunks = chunk_text(document)

    # Create embeddings
    embedding_model = create_embedding_model()
    embeddings = embedding_model.encode(chunks, convert_to_tensor=True)

    # Connect to Neo4j
    neo4j_uri = ("bolt://18.213.110.87")
    neo4j_user =  "neo4j"
    neo4j_password = ("increments-floats-tours")
    driver = connect_to_neo4j(neo4j_uri, neo4j_user, neo4j_password)

    # Add chunks to Neo4j
    pdf_name = os.path.basename(file_path)
    add_chunks_to_neo4j(driver, chunks, embeddings, pdf_name)

    # Query loop
    while True:
        query = input("Enter your query (or type 'exit' to quit): ")
        if query.lower() == 'exit':
            break

        top_chunks = query_neo4j(driver, query, embedding_model, chunks, embeddings, pdf_name)
        print("\nTop relevant chunks:")
        for idx, chunk, score in top_chunks:
            print(f"Chunk {idx + 1} (Score: {score:.4f}):\n{chunk}\n")

    # Close Neo4j connection
    driver.close()
    print("Process complete!")

if __name__ == "__main__":
    main()
