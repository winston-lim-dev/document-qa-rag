from sentence_transformers import SentenceTransformer
import chromadb

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

client = chromadb.PersistentClient(
    path="chroma_db"
)

collection = client.get_or_create_collection(
    name="documents"
)

documents = [
    "Microsoft develops Windows.",
    "Apple develops iPhones.",
    "OpenAI develops ChatGPT."
]

for i, doc in enumerate(documents):

    embedding = model.encode(doc).tolist()

    collection.add(
        ids=[str(i)],
        documents=[doc],
        embeddings=[embedding]
    )

print("Documents stored.")