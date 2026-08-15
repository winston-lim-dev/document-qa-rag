import streamlit as st
from sentence_transformers import SentenceTransformer
from ollama import chat
import chromadb

from src.document_qa.ingestion import IngestionError, ingest_pdf


# --------------------------------------
# Load embedding model once
# --------------------------------------
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


model = load_model()


# --------------------------------------
# ChromaDB persistent storage
# --------------------------------------
client = chromadb.PersistentClient(
    path="chroma_db"
)

collection = client.get_or_create_collection(
    name="documents"
)


# --------------------------------------
# UI
# --------------------------------------
st.title("📄 Winston's Local RAG Assistant")

st.write(
    "Upload a PDF and ask questions about it."
)

uploaded_file = st.file_uploader(
    "Upload PDF",
    type="pdf"
)

question = st.text_input(
    "Ask a question about the document"
)


# --------------------------------------
# PDF Processing
# --------------------------------------
if uploaded_file:
    try:
        all_chunks = ingest_pdf(uploaded_file, uploaded_file.name)
    except IngestionError as error:
        st.error(str(error))
        st.stop()

    st.success("PDF loaded successfully")

    # Create embeddings
    # --------------------------------------
    
    documents = []
    metadatas = []
    embeddings = []

    for chunk_data in all_chunks:

        documents.append(
            chunk_data.text
        )

        metadatas.append(
            {
                "page": chunk_data.page,
                "document_id": chunk_data.document_id,
                "filename": chunk_data.filename,
                "chunk_id": chunk_data.chunk_id,
            }
        )

        embedding = model.encode(
            chunk_data.text
        ).tolist()

        embeddings.append(embedding)
        
    # --------------------------------------
    # Clear previous data
    # --------------------------------------
    existing = collection.get()

    if len(existing["ids"]) > 0:
        collection.delete(
            ids=existing["ids"]
        )

    # --------------------------------------
    # Store in ChromaDB
    # --------------------------------------
    ids = [chunk.chunk_id for chunk in all_chunks]

    collection.add(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas
    )

    st.success("Document indexed")

    # --------------------------------------
    # Ask Question
    # --------------------------------------
    if question:

        query_embedding = model.encode(
            question
        ).tolist()

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=3
        )

        retrieved_chunks = results["documents"][0]

        context = "\n\n".join(
            retrieved_chunks
        )


        prompt = f"""
You are a helpful document assistant.

Use the retrieved document context to answer the question.

If the answer is partially available,
provide the best answer you can.

Context:
{context}

Question:
{question}

Answer:
"""
        with st.spinner(
            "Generating answer..."
        ):

            response = chat(
                model="llama3.2:3b",
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )

            answer = (
                response["message"]["content"]
            )

        st.subheader("Answer")
        st.write(answer)

        st.write("Sources")

        retrieved_metadata = results["metadatas"][0]

        pages = set()

        for metadata in retrieved_metadata:
            pages.add(metadata["page"])

        st.subheader("Sources")

        for page in sorted(pages):
            st.write(f"Page {page}")
       
        with st.expander(
            "View Retrieved Chunks"
        ):

            for i, chunk in enumerate(
                retrieved_chunks,
                start=1
            ):

                st.markdown(
                    f"### Chunk {i}"
                )

                st.write(chunk)
