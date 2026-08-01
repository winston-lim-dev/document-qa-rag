import streamlit as st
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from ollama import chat
import chromadb


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

    pdf = PdfReader(uploaded_file)

    text = ""

    for page in pdf.pages:
        page_text = page.extract_text()

        if page_text:
            text += page_text

    st.success("PDF loaded successfully")

    st.write(
        f"Document length: {len(text)} characters"
    )

    # --------------------------------------
    # Chunk document
    # --------------------------------------
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = splitter.split_text(text)

    st.write(
        f"Chunks created: {len(chunks)}"
    )

    # --------------------------------------
    # Create embeddings
    # --------------------------------------
    embeddings = []

    for chunk in chunks:

        embedding = model.encode(
            chunk
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
    ids = [str(i) for i in range(len(chunks))]

    collection.add(
        ids=ids,
        documents=chunks,
        embeddings=embeddings
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
Answer the question using ONLY the
information provided below.

If the answer is not found in the context,
say "I could not find the answer in the document."

Context:
{context}

Question:
{question}
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