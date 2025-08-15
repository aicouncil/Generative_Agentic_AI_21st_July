# RAG-based Chatbot with Gradio UI

This document provides a detailed explanation of the `rag_gradio_g3.py` script, which implements a Retrieval-Augmented Generation (RAG) system with an interactive user interface built using the `gradio` library. The application allows a user to upload a PDF file and then ask questions whose answers are grounded in the content of that document.

***

## 1. Introduction to RAG and Gradio

* **Retrieval-Augmented Generation (RAG):** RAG is a technique that enhances the capabilities of a generative Large Language Model (LLM) by providing it with external, relevant information from a knowledge base to generate more accurate and grounded responses. Instead of relying solely on the LLM's pre-trained knowledge, a RAG system first retrieves relevant documents or data chunks and then uses that context to formulate the final answer.
* **Gradio:** Gradio is an open-source Python library that simplifies the process of creating user interfaces for machine learning models. It automatically generates a web-based UI from a Python script, which is ideal for quickly demonstrating machine learning applications.

***

## 2. RAG Pipeline Workflow

The script defines a complete RAG pipeline by chaining several components from the `langchain` and `langchain-community` libraries.

### 2.1. Data Ingestion and Preprocessing

* **PDF Text Extraction:** The `PyPDF2` library is used to open a PDF file in binary read mode and extract its text content.
    ```python
    def extract_data_from_pdf(pdf_path):
        with open(pdf_path , 'rb') as file:
            pdfreader = PyPDF2.PdfReader(file)
            full_text = ''
            for page in pdfreader.pages:
                full_text += page.extract_text()
        return full_text
    ```
* **Text Splitting:** The large block of text extracted from the PDF is split into smaller, more manageable chunks to fit within the context window of an LLM. The `RecursiveCharacterTextSplitter` is used for this with specific parameters to define the chunk size and overlap.
    * `chunk_size=300`: Each chunk is a maximum of 300 characters long.
    * `chunk_overlap=100`: The chunks overlap by 100 characters to ensure that context is maintained between them.
    ```python
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    def split_text(text):
      splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=100)
      docs = splitter.create_documents([text])
      return docs
    ```

### 2.2. Vectorization and Retrieval Setup

* **Vector Store Creation:** This step converts the text chunks into numerical vector representations (embeddings) that a computer can understand.
    * **Embeddings:** `OpenAIEmbeddings` is used to transform each text chunk into a high-dimensional vector that captures its semantic meaning.
    * **FAISS:** The generated embeddings are stored in a `FAISS` vector store. FAISS is a library that enables fast and efficient similarity search, which is crucial for quickly retrieving relevant text chunks for a given query.
    ```python
    from langchain_community.embeddings import OpenAIEmbeddings
    from langchain_community.vectorstores import FAISS
    def create_vector_store(docs):
      embeddings = OpenAIEmbeddings()
      vectorstore = FAISS.from_documents(docs , embeddings)
      return vectorstore
    ```
* **RAG Chain Assembly:** This function assembles the RAG components to create a functional question-answering system.
    * **Retriever:** The `vectorstore.as_retriever()` component is defined to search the vector store for the text chunks with the highest semantic similarity to the user's query.
    * **LLM:** The `ChatGoogleGenerativeAI` model (`gemini-2.5-flash`) is specified as the generative model that will synthesize the final answer.
    * **`RetrievalQA` Chain:** The `RetrievalQA.from_chain_type` function combines the LLM and the retriever into a single, executable chain.
    ```python
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain.chains import RetrievalQA
    def setup_rag_qa(vectorstore):
      retriever = vectorstore.as_retriever(search_type = 'similarity')
      llm = ChatGoogleGenerativeAI(model = "gemini-2.5-flash")
      rag_chain = RetrievalQA.from_chain_type(llm=llm , retriever=retriever)
      return rag_chain
    ```

### 3. Gradio User Interface and Interaction Logic

The script uses `gradio` to build a simple and user-friendly interface that ties all the RAG components together.

* **UI Layout:** The UI is defined within a `gr.Blocks()` context manager, which allows for custom layouts. `gr.Markdown` is used for titles, `gr.File` for uploading a PDF, and `gr.Textbox` for displaying status, accepting questions, and showing answers. `gr.Row` is used to organize components horizontally.
* **Interaction Functions:**
    * **`upload_pdf(file)`:** This function is triggered when a user uploads a PDF. It executes the entire RAG pipeline (extracts text, splits it, creates the vector store, and sets up the RAG chain). It returns a status message to the UI.
    * **`ask_question(query)`:** This function is triggered when the user submits a question. It passes the question to the `qa_chain` and returns the generated answer.
* **Event Handling:**
    * The `.change()` method is used on the `gr.File` component to automatically trigger the `upload_pdf` function whenever a new PDF is uploaded.
    * The `.submit()` method on the `gr.Textbox` is used to trigger the `ask_question` function when the user presses Enter or clicks a submit button.
    ```python
    import gradio as gr
    # The functions and global variables are defined above this block.

    with gr.Blocks() as ui_demo:
      gr.Markdown("# RAG assistant with GPT")
      gr.Markdown("Upload a PDF, Then ask any questions from its content.")

      with gr.Row():
        pdf_input = gr.File(label = "Upload PDF")
        upload_status = gr.Textbox(label = "Upload Status")

      pdf_input.change(fn = upload_pdf , inputs = pdf_input , outputs=upload_status)

      with gr.Row():
        question_input = gr.Textbox(label="Ask few questions")
        answer_output = gr.Textbox(label="Answer")

      question_input.submit(fn=ask_question , inputs = question_input , outputs = answer_output)

    ui_demo.launch()
    ```
* **Use Cases and Benefits:** This application serves as a versatile tool for question answering based on a user-provided document. It can be used for customer support, enterprise search, and educational purposes. The main benefit is that it combines the power of LLMs with specific data sources, making the system practical and robust.