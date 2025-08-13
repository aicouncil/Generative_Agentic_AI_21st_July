# Detailed Explanation of `genai_g2_d13(rag_gpt).ipynb`  
*Repo: [aicouncil/Generative_Agentic_AI_21st_July](https://github.com/aicouncil/Generative_Agentic_AI_21st_July)*  
*File: day_13/genai_g2_d13(rag_gpt).ipynb*

---

## 1. Overview

This notebook demonstrates **Retrieval Augmented Generation (RAG)** using GPT and Google Gemini models, focusing on how to enhance a language model's responses by providing it with relevant, contextually retrieved data (e.g., from a PDF document). The workflow automates the process of extracting information from long documents, chunking it, embedding it, storing it in a vector database, and then querying it using an LLM (Large Language Model).

---

## 2. Key Concepts Covered

### 2.1. Retrieval Augmented Generation (RAG)

- **Definition:** RAG is a technique where a language model's output is improved by supplementing its internal knowledge with retrieved information from an external source (such as documents, databases, or the web).
- **Steps:**
  1. **Retrieve:** Relevant data chunks are fetched from a knowledge base in response to a user's query.
  2. **Augment:** These chunks are provided to the language model as additional context.
  3. **Generate:** The language model generates a response, leveraging both its own knowledge and the retrieved context.

**Example Use Cases:**
- Automated Q&A over company documents or manuals.
- Building smart chatbots that can answer questions about policies, procedures, or technical documentation.
- Customer support agents that can search and answer queries from product guides.

---

### 2.2. Workflow Steps in the Notebook

#### Step 1: Install Required Libraries

The notebook installs the following packages:
- `PyPDF2` for PDF extraction
- `langchain-community` and `langchain_google_genai` for LLM and RAG pipeline
- `faiss-cpu` for vector storage and similarity search

#### Step 2: Import Libraries

- Imports modules for PDF reading, embeddings, vector database (FAISS), text splitting, and LLMs (OpenAI, Gemini).

#### Step 3: PDF Data Extraction

```python
def extract_data_from_pdf(pdf_path):
    with open(pdf_path , 'rb') as file:
        pdfreader = PyPDF2.PdfReader(file)
        full_text = ''
        for page in pdfreader.pages:
            full_text += page.extract_text()
    return full_text
```
**Explanation:** This function reads all text from a provided PDF, enabling further processing.

**Use Case:** Ingesting user manuals, legal documents, research papers for downstream Q&A.

#### Step 4: Text Splitting

```python
def split_text(text):
  splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=100)
  docs = splitter.create_documents([text])
  return docs
```
**Explanation:** Long texts are split into manageable chunks (e.g., 300 characters with overlap) for embedding and retrieval.

**Example:** Splitting a 100-page manual into smaller sections, so questions can be answered based on relevant sections.

#### Step 5: Create Vector Store

```python
def create_vector_store(docs):
  embeddings = OpenAIEmbeddings()
  vectorstore = FAISS.from_documents(docs , embeddings)
  return vectorstore
```
**Explanation:** Each text chunk is embedded (converted to a vector), then stored in FAISS (a vector database), enabling fast similarity search.

**Example:** When a user asks a question, the system finds the most relevant chunks using vector similarity.

#### Step 6: Setup RAG QA Pipeline

```python
def setup_rag_qa(vectorstore):
  retriever = vectorstore.as_retriever(search_type = 'similarity')
  llm = ChatGoogleGenerativeAI(model = "gemini-2.5-flash")
  rag_chain = RetrievalQA.from_chain_type(llm=llm , retriever=retriever)
  return rag_chain
```
**Explanation:** Combines the retriever (similarity search over the vector store) with an LLM (Gemini or GPT) in a chain. When a query is issued, the retriever finds the most relevant document chunks; the LLM answers using both retrieved data and its own knowledge.

**Use Case:** Answering specific, context-sensitive questions from large document collections.

#### Step 7: Complete Pipeline Example

```python
# Environment setup for API keys (from Google Colab's userdata)
pdf_path = '/content/company_manual.pdf'
text = extract_data_from_pdf(pdf_path)
docs = split_text(text)
vectorstore = create_vector_store(docs)
qa_chain = setup_rag_qa(vectorstore)

# Example query
query = "Tell me about the company and the product, The output expected in bullet form. Also want to know the brief of warranty"
result = qa_chain(query)
print(result['result'])
```
**Explanation:** This sequence processes a company manual and allows users to query it, getting detailed, context-rich answers.

---

## 3. Example Queries & Outputs

### Example 1: Company and Warranty Information

**Query:**  
"Tell me about the company and the product, output expected in bullet form. Also want to know the brief of warranty"

**Output:**  
- Lists company background, product info, and warranty details in bullets.
- Illustrates how structured answers can be generated from unstructured documents.

### Example 2: Customer Support Contact (JSON format)

**Query:**  
"How to reach customer support, output format - json"

**Output:**  
```json
{
  "customer_support_channels": [
    {"type": "Phone", "number": "+91-9999999999", "hours": "Mon-Fri, 9 AM to 6 PM IST"},
    {"type": "Email", "address": "support@technova.com", "response_time": "within 24 hours"},
    {"type": "Live Chat", "availability": "Available on our website and mobile app"},
    {"type": "Help Center", "url": "https://support.technova.com"},
    {"type": "Video Call Support", "note": "for troubleshooting smart home devices"},
    {"type": "Remote Diagnostics", "note": "for troubleshooting smart home devices"}
  ],
  "general_response_aim": "resolve most customer queries within 48 hours"
}
```
- Demonstrates how output can be formatted as JSON for integration with other systems.

---

## 4. Practical Use Cases

### 4.1. Enterprise Document Q&A

- **Scenario:** Employees or customers query policies, manuals, contracts, or technical documentation.
- **Benefit:** Quick, accurate answers without manual document search.

### 4.2. Customer Support Automation

- **Scenario:** Chatbots answer support queries by searching through manuals, troubleshooting guides, and FAQs.
- **Benefit:** Reduces support workload and response time.

### 4.3. Compliance & Regulatory Q&A

- **Scenario:** Legal teams or auditors query compliance documents for specific clauses or procedures.
- **Benefit:** Ensures accurate, context-driven responses for audits.

### 4.4. Educational Tools

- **Scenario:** Students or researchers query textbooks, research papers, or course materials.
- **Benefit:** Enhances learning by providing instant, relevant answers from source materials.

---

## 5. Summary

This notebook is a practical guide to implementing **RAG (Retrieval Augmented Generation)** using popular LLMs (GPT, Gemini) and open-source tools (`langchain`, `FAISS`). It demonstrates:
- How to ingest and process large documents.
- How to enable semantic search and context-rich Q&A over those documents.
- How to integrate with modern LLMs for natural, informative answers.

**The workflow is highly extensible:**  
- Can be adapted for different data sources (web pages, databases, spreadsheets).
- Can be fine-tuned for domain-specific applications (medical, legal, technical).

---

## 6. References

- [LangChain Documentation](https://python.langchain.com/docs/)
- [FAISS: Facebook AI Similarity Search](https://github.com/facebookresearch/faiss)
- [Google Gemini](https://deepmind.google/technologies/gemini/)
- [OpenAI GPT](https://openai.com/gpt-4)
- [PyPDF2 Documentation](https://pypdf2.readthedocs.io/en/latest/)
- [RAG Paper (Facebook Research)](https://ai.facebook.com/blog/-retrieval-augmented-generation/)
