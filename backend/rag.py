from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
import os
from pinecone import Pinecone as PineconeClient, ServerlessSpec
from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv
load_dotenv()

def load_pdf_files_corrected(data_path):
    all_documents = []
    total_pages = 0
    
    # Get all PDF files
    pdf_files = [f for f in os.listdir(data_path) if f.endswith('.pdf')]
    
    for pdf_file in pdf_files:
        print(f"Processing: {pdf_file}")
        loader = PyPDFLoader(os.path.join(data_path, pdf_file))
        documents = loader.load()
        all_documents.extend(documents)
        print(f"Pages in {pdf_file}: {len(documents)}")
        total_pages += len(documents)
        
        print(f"Total pages across all PDFs: {total_pages}")
    return all_documents


documents = load_pdf_files_corrected('../rag/')
print(f"Final document count: {len(documents)}")

def splitter(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(text)
    return chunks

text = splitter(documents)
len(text)    

api_key = os.getenv("PINECONE_API_KEY")

index_name = "retinopathy"

pc = PineconeClient(api_key=api_key)

pc.create_index(
    name=index_name,
    dimension=384,  
    metric="cosine",
    spec=ServerlessSpec(
        cloud='aws',
        region='us-east-1'
    )
)

index = pc.Index(index_name)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


from langchain_pinecone import PineconeVectorStore
vector_store = PineconeVectorStore.from_documents(
    documents=text,
    embedding=embeddings,
    index_name=index_name
)