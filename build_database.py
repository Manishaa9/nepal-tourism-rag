from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

print("1. Loading the document...")
loader = TextLoader("nepal_tourism.txt", encoding="utf-8")
documents = loader.load()

print("2. Splitting into chunks...")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(documents)
print(f"Created {len(chunks)} chunks.")

print("3. Creating embeddings (this may take a moment)...")
embeddings = OllamaEmbeddings(model="nomic-embed-text")

print("4. Saving to ChromaDB...")
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

print("Success! Database built and saved.")