# Retrieval-Augmented Generation (RAG) Application with Streamlit

This document provides a detailed explanation of the `rag_app.py` script, which implements a Retrieval-Augmented Generation (RAG) system as an interactive chatbot using the Streamlit framework. The application's purpose is to answer user queries based on the factual content of a specific PDF document.

## 1. Introduction to Retrieval-Augmented Generation (RAG)

**Definition:** RAG is a technique that enhances the capabilities of a Large Language Model (LLM) by providing it with external, relevant information from a knowledge base to generate more accurate and grounded responses. Instead of relying solely on the data it was pre-trained on, a RAG system first **retrieves** relevant documents or data chunks and then uses that context to **generate** the final answer.

* **Benefits:**
    * **Reduces Hallucinations:** By grounding the LLM in specific, verifiable information, RAG significantly reduces the model's tendency to "hallucinate" or generate factually incorrect information.
    * **Enables Proprietary Knowledge:** It allows companies to use LLMs on their private or up-to-date documents (like a company manual) without retraining the entire model.
    * **Improves Accuracy:** Responses are directly supported by the provided context.

## 2. The RAG Pipeline Workflow

The script builds a complete RAG pipeline by chaining several components from the `langchain` and `langchain-community` libraries.

### 2.1. Data Extraction and Preprocessing

* **PDF Extraction (`extract_data_from_pdf`):**
    * The `PyPDF2` library is used to read text from a PDF file located at a predefined path (`PDF_PATH = 'D:/RAG_UI/company_manual.pdf'`).
    * The function opens the file, creates a `PdfReader` object, and iterates through every page to extract and concatenate the text into a single string.
* **Text Splitting (`split_text`):**
    * The large block of text extracted from the PDF is too long to fit into an LLM's context window. The `RecursiveCharacterTextSplitter` from `langchain` is used to break the text into smaller, more manageable chunks.
    * The key parameters are:
        * `chunk_size=300`: Each chunk of text will be approximately 300 characters long.
        * `chunk_overlap=100`: The chunks will overlap by 100 characters. This ensures that the context from the end of one chunk is carried over to the beginning of the next, preventing loss of information at the boundaries.

### 2.2. Vectorization and Retrieval

* **Vector Store Creation (`create_vector_store`):**
    * A vector store is a database designed to store and search for numerical vector representations of data. This step converts the text chunks into these vectors.
    * **Embeddings:** `OpenAIEmbeddings` is used to transform each text chunk into a high-dimensional vector, capturing its semantic meaning.
    * **Vector Store:** The generated embeddings are stored in a `FAISS` vector store. FAISS is a library for efficient similarity search and clustering of dense vectors, enabling fast retrieval of relevant chunks.
* **RAG Chain Setup (`setup_rag_qa`):**
    * This function assembles the RAG components to create the final question-answering system.
    * **Retriever:** `vectorstore.as_retriever(search_type = 'similarity')` is defined. This component is responsible for taking a user's query, converting it to an embedding, and searching the `FAISS` vector store for the text chunks with the highest semantic similarity.
    * **Large Language Model (LLM):** `ChatGoogleGenerativeAI(model = "gemini-2.5-flash")` is specified as the generative model. The LLM's role is to synthesize the retrieved context and the user's original query into a coherent and human-like response.
    * **RetrievalQA Chain:** The `RetrievalQA.from_chain_type` function combines the LLM and the retriever into a single, executable chain. When this chain is called, it automatically performs the retrieval, feeds the results to the LLM, and returns the final answer.

## 3. Streamlit UI Integration

The script uses Streamlit to create a simple and user-friendly interface for the RAG chatbot.

* **API Key Configuration:** The script sets environment variables for the `OPENAI_API_KEY` and `GOOGLE_API_KEY`.
* **Caching the Pipeline (`@st.cache_resource`):**
    The `@st.cache_resource` decorator is applied to the `load_pipeline()` function. This is a crucial optimization in Streamlit. It ensures that the computationally expensive steps of extracting text, splitting, and creating the vector store are executed only once when the app is first run, and the result is cached. Subsequent interactions will use the cached `qa_chain` object, making the application highly responsive.
* **User Interaction:**
    * `st.title` sets the title of the web app.
    * `st.text_input` creates a text box where the user can type their query.
    * `st.button` creates a button to trigger the search.
    * When the button is clicked and a query is present, the `qa_chain(query)` is executed. The final answer is then extracted and displayed to the user using `st.write(result['result'])`.
    ```python
    st.title("Customer Assistant Bot-")
    qa_chain = load_pipeline()
    query = st.text_input("Ask your query about product or company:",)
    if st.button("Get Answer") and query:
       result = qa_chain(query)
       st.write(result['result'])
    ```
* **Use Cases:**
    * **Customer Support:** Creating a chatbot that can answer customer questions based on a knowledge base of documents.
    * **Enterprise Search:** Enabling employees to query internal documents, manuals, and reports using natural language.
    * **Education:** Building question-answering systems for educational materials or research papers.
