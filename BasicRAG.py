import os
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import PointStruct
from langchain_openai import OpenAIEmbeddings

# Set OpenAI API key in environment or pass explicitly
os.environ["OPENAI_API_KEY"] = "YOUR_API_KEY"

# Initialize Qdrant client with full URL
client = QdrantClient(url="http://localhost:6333")

# Initialize embedding model
embedding_model = OpenAIEmbeddings(model="text-embedding-ada-002")

texts = [
    "Open source vector databases allow self-hosted similarity search.",
    "LLMs generate vector embeddings for semantic search.",
    "Qdrant provides filtering and payload support with vectors."
]

# Generate embeddings
embeddings = [embedding_model.embed_query(text) for text in texts]

# Prepare points for Qdrant
points = [
    PointStruct(id=i, vector=embedding, payload={"text": text})
    for i, (embedding, text) in enumerate(zip(embeddings, texts))
]

# Create collection if not exists
collection_name = "example_collection"

if collection_name not in client.get_collections().collections:
    client.recreate_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(
            size=len(embeddings[0]),
            distance=models.Distance.COSINE
        )
    )

# Upsert points
client.upsert(collection_name=collection_name, points=points)

# Query example
query_text = "How do vector embeddings help heuristic search?"
query_embedding = embedding_model.embed_query(query_text)

results = client.search(collection_name=collection_name, query_vector=query_embedding, limit=3)

for res in results:
    print(f"ID: {res.id}, Score: {res.score}, Text: {res.payload['text']}")
