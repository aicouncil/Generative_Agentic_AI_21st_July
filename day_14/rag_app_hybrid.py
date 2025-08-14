import streamlit as st
import os
import PyPDF2
from openai import OpenAI
import google.generativeai as genai
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.chat_models import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA

PDF_PATH = 'D:/RAG_UI/company_manual.pdf'

def extract_data_from_pdf(pdf_path):
    with open(pdf_path , 'rb') as file:
        pdfreader = PyPDF2.PdfReader(file)
        full_text = ''
        for page in pdfreader.pages:
            full_text += page.extract_text()
    return full_text

def split_text(text):
  splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=100)
  docs = splitter.create_documents([text])
  return docs

def create_vector_store(docs):
  embeddings = OpenAIEmbeddings()
  vectorstore = FAISS.from_documents(docs , embeddings)
  return vectorstore

def setup_rag_qa(vectorstore):
  retriever = vectorstore.as_retriever(search_type = 'similarity')
  #llm = ChatOpenAI(model = "gpt-4.1-nano")
  llm = ChatGoogleGenerativeAI(model = "gemini-2.5-flash")
  rag_chain = RetrievalQA.from_chain_type(llm=llm , retriever=retriever)
  return rag_chain

#-------------------------
# Streamlit UI
#-----------------------------

st.title("Customer Assistant Bot-")


os.environ['OPENAI_API_KEY'] = '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'
os.environ['GOOGLE_API_KEY'] = 'AIzaSyAJoYFwz7rFEMHKSr59uHtP6ikF4TOp5tU'

@st.cache_resource
def load_pipeline():
    text = extract_data_from_pdf(PDF_PATH)
    docs = split_text(text)
    vectorstore = create_vector_store(docs)
    return setup_rag_qa(vectorstore)

qa_chain = load_pipeline()

#User query
query = st.text_input(
    "Ask your query about product or company:",
)

def hybrid_answer(qa_chain, query):
    result = qa_chain(query)
    answer = result['result']

    if "I don't know" in answer or "not mention" in answer.lower():
        # Fallback to general LLM without retrieval
        from langchain_google_genai import ChatGoogleGenerativeAI
        llm_general = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
        answer = llm_general.invoke(query).content
    
    return answer

if st.button("Get Answer") and query:
   result = hybrid_answer(qa_chain, query)
   st.write(result)

   