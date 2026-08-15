import streamlit as st
from ollama import chat

from src.document_qa.ingestion import IngestionError, ingest_pdf
from src.document_qa.retrieval import DocumentRetriever


# --------------------------------------
# Load retrieval resources once
# --------------------------------------
@st.cache_resource
def load_retriever():
    return DocumentRetriever.persistent()


retriever = load_retriever()


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

    retriever.index(all_chunks)

    st.success("Document indexed")

    # --------------------------------------
    # Ask Question
    # --------------------------------------
    if question:

        retrieval_results = retriever.retrieve(question, top_k=3)
        retrieved_chunks = [result.chunk.text for result in retrieval_results]

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

        pages = {result.chunk.page for result in retrieval_results}

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
