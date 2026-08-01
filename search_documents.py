from sentence_transformers import SentenceTransformer
import chromadb

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

client = chromadb.PersistentClient(
    path="chroma_db"
)

collection = client.get_collection(
    "documents"
)

query = "Who made iphone 15 ?"

embedding = model.encode(query).tolist()

results = collection.query(
    query_embeddings=[embedding],
    n_results=1
)

print(results["documents"])
