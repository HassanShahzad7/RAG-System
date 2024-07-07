# Import necessary libraries
import fitz  # PyMuPDF for PDF processing
import re  # Regular expressions for text extraction
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Function to parse the PDF document and return the text content
def parse_pdf(file_path):
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# Function to extract information from the parsed text
def extract_info(text):
    papers = []
    paper_pattern = re.compile(r"Title: (.*?)\nAuthors: (.*?)\nDate: (.*?)\nDescription: (.*?)\nStats: (.*?)\nCategories: (.*?)\nLinks: (.*?)\n", re.DOTALL)
    matches = paper_pattern.findall(text)
    for match in matches:
        papers.append({
            "title": match[0].strip(),
            "authors": match[1].strip(),
            "date": match[2].strip(),
            "description": match[3].strip(),
            "stats": match[4].strip(),
            "categories": match[5].strip(),
            "links": match[6].strip()
        })
    return papers

# Load the model and tokenizer
model_name = "EleutherAI/gpt-j-6B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Define the document loader
def load_documents(file_path):
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    return documents

# Define the text splitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

# Define the embeddings
embeddings = HuggingFaceEmbeddings()

# Define the vector store
def create_vector_store(documents):
    texts = text_splitter.split_documents(documents)
    vector_store = FAISS.from_documents(texts, embeddings)
    return vector_store

# Define the RetrievalQA chain
def create_retrieval_qa_chain(vector_store):
    retriever = vector_store.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(llm=model, retriever=retriever, return_source_documents=True)
    return qa_chain

# Function to load the PDF, process the data, and answer the query
def load_and_process_pdf(file_path, query):
    # Load and parse PDF
    pdf_text = parse_pdf(file_path)
    papers_info = extract_info(pdf_text)
    
    # Load documents
    documents = load_documents(file_path)
    
    # Create vector store
    vector_store = create_vector_store(documents)
    
    # Create RetrievalQA chain
    qa_chain = create_retrieval_qa_chain(vector_store)
    
    # Generate response
    response = qa_chain.run(query)
    
    return response

# Example usage
file_path = "RAG Input Doc.pdf"
query = "Which paper received the highest number of stars per hour?"
response = load_and_process_pdf(file_path, query)
print(response)