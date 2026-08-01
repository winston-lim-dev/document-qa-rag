from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load PDF
pdf = PdfReader("data/sample1.pdf")

# Extract text
text = ""

for page in pdf.pages:
    page_text = page.extract_text()
    if page_text:
        text += page_text

print(f"Characters extracted: {len(text)}")

# Split into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

chunks = splitter.split_text(text)

print(f"Number of chunks: {len(chunks)}")

print("\nFirst chunk:\n")
print(chunks[0])

