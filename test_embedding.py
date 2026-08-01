from sentence_transformers import SentenceTransformer

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

embedding = model.encode(
    "Artificial Intelligence is changing business."
)

print(len(embedding))
print(embedding[:5])