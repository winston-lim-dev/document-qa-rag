from langchain_text_splitters import RecursiveCharacterTextSplitter

text = """
Artificial Intelligence is transforming industries.
Python is widely used in AI development.
Large Language Models are becoming common.
"""

splitter = RecursiveCharacterTextSplitter(
    chunk_size=30,
    chunk_overlap=2
)

chunks = splitter.split_text(text)

for chunk in chunks:
    print(chunk)