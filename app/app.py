from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

embeddings = OllamaEmbeddings(model="llama3")
vector_db = Chroma(embedding_function=embeddings, collection_name="pdf_collection", persist_directory="./chroma_db")

@app.route('/upload', methods=['POST'])
def upload_pdf():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    file_path = f"./uploads/{file.filename}"
    file.save(file_path)  # Save the uploaded file

    # Load and add the PDF to the database
    add_pdf_to_db(file_path)

    return jsonify({'message': 'PDF uploaded and processed successfully.'})

def add_pdf_to_db(file_path):
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    chunks = text_splitter.split_documents(documents)

    vector_db.add_documents(chunks)
    print(f"Added {len(chunks)} chunks to the database.")

@app.route('/query', methods=['POST'])
def query_embedding():
    data = request.json
    message = data.get('message', '')
    if not message:
        return jsonify({'error': 'No query provided'}), 400

    # Implement logic to retrieve from Chroma vector_db based on the message
    response = f"You asked: {message}. (This is a placeholder response.)"
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True)
