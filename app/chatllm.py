from langchain_ollama import ChatOllama
from langchain_chroma import Chroma
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.embeddings import OllamaEmbeddings

# Initialize LLM and database
llm = ChatOllama(model="llama3", temperature=0.7)
vector_db = Chroma(embedding_function=OllamaEmbeddings(model="llama3"), collection_name="pdf_collection", persist_directory="./chroma_db")

# Set up the retriever
retriever = ContextualCompressionRetriever(
    base_compressor=LLMChainExtractor.from_llm(llm),
    base_retriever=vector_db.as_retriever()
)

# Define prompt and conversational chain
prompt_template = ChatPromptTemplate.from_template("""
    Context: {context}
    Human: {question}
    AI: Please answer based on the provided context.
""")

conversation_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=ConversationBufferMemory(memory_key="chat_history", return_messages=True),
    combine_docs_chain_kwargs={"prompt": prompt_template}
)

def chatbot_response(query):
    response = conversation_chain({"question": query})
    return response["answer"]
