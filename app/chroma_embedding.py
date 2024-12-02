# Use PyPDFLoader to load the PDF, and Chroma to store the embeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

embeddings = OllamaEmbeddings(model="llama3")
vector_db = Chroma(embedding_function=embeddings, collection_name="pdf_collection", persist_directory="./chroma_db")

def add_pdf_to_db(file_path):
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    chunks = text_splitter.split_documents(documents)
    
    vector_db.add_documents(chunks)
    print(f"Added {len(chunks)} chunks to the database.")

# Run this function to add a PDF to the database
add_pdf_to_db("path/to/your/file.pdf")
