import streamlit as st

from src.document_qa.generation import OllamaGenerator
from src.document_qa.ingestion import IngestionError, ingest_pdf
from src.document_qa.retrieval import DocumentRetriever
from src.document_qa.service import DocumentQAService


@st.cache_resource
def load_retriever() -> DocumentRetriever:
    """Load the embedding model and persistent collection once per app process."""
    return DocumentRetriever.persistent()


st.title("📄 Winston's Local RAG Assistant")
st.write("Upload a PDF and ask questions about it.")

try:
    retriever = load_retriever()
except Exception as error:
    st.error(f"Unable to initialize local retrieval: {error}")
    st.stop()

qa_service = DocumentQAService(retriever, OllamaGenerator())

uploaded_file = st.file_uploader("Upload PDF", type="pdf")
question = st.text_input("Ask a question about the document")

if uploaded_file:
    try:
        chunks = ingest_pdf(uploaded_file, uploaded_file.name)
    except IngestionError as error:
        st.error(f"This PDF cannot be indexed: {error}")
        st.stop()
    except Exception as error:
        st.error(f"Unable to read this PDF: {error}")
        st.stop()

    st.success("PDF loaded successfully")

    document_id = chunks[0].document_id
    if st.session_state.get("indexed_document_id") != document_id:
        try:
            retriever.index(chunks)
        except Exception as error:
            st.error(f"Unable to index this document: {error}")
            st.stop()
        st.session_state["indexed_document_id"] = document_id
        st.success("Document indexed")
    else:
        st.caption("Document already indexed for this session.")

    if question:
        try:
            with st.spinner("Retrieving evidence and generating answer..."):
                result = qa_service.answer(question, top_k=3)
        except Exception as error:
            st.error(
                "Unable to answer the question. Confirm that Ollama is running "
                f"and the local models are available: {error}"
            )
            st.stop()

        st.subheader("Answer")
        st.write(result.answer)
        if not result.has_sufficient_evidence:
            st.info("No usable retrieval evidence was available for generation.")

        st.subheader("Sources")
        st.caption("Chroma distance is shown as returned; lower means closer.")
        for index, evidence in enumerate(result.evidence, start=1):
            st.write(
                f"[S{index}] {evidence.chunk.filename} — Page "
                f"{evidence.chunk.page} — Distance {evidence.distance:.4f}"
            )

        if not result.evidence:
            st.caption("No sources to display.")

        with st.expander("View retrieved evidence"):
            for index, evidence in enumerate(result.evidence, start=1):
                st.markdown(
                    f"### [S{index}] {evidence.chunk.filename}, "
                    f"page {evidence.chunk.page}"
                )
                st.write(evidence.chunk.text)
