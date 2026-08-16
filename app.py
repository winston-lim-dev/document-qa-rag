import streamlit as st

from src.document_qa.generation import OllamaGenerator
from src.document_qa.ingestion import IngestionError, ingest_pdf
from src.document_qa.retrieval import DocumentRetriever
from src.document_qa.service import DocumentQAService


# --------------------------------------
# Load retrieval resources once
# --------------------------------------
@st.cache_resource
def load_retriever():
    return DocumentRetriever.persistent()


retriever = load_retriever()
qa_service = DocumentQAService(retriever, OllamaGenerator())


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
        with st.spinner(
            "Generating answer..."
        ):
            result = qa_service.answer(question, top_k=3)

        st.subheader("Answer")
        st.write(result.answer)

        st.subheader("Sources")

        sources = {
            (evidence.chunk.filename, evidence.chunk.page)
            for evidence in result.evidence
        }
        for filename, page in sorted(sources):
            st.write(f"{filename} — Page {page}")
       
        with st.expander(
            "View Retrieved Chunks"
        ):

            for i, evidence in enumerate(
                result.evidence,
                start=1
            ):

                st.markdown(
                    f"### Chunk {i}"
                )

                st.write(evidence.chunk.text)
