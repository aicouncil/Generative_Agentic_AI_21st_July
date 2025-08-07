This file is a summary of the "Generative & Agentic AI - 22 July" meeting from August 6, 2025. It details the process of implementing a semantic search chatbot, as explained by AI Council.

Key aspects covered in the meeting include:

  * **Installation of Dependencies:** Using `sentence-transformer` for semantic embedding and `PI PDF2` for PDF data extraction.
  * **Text Extraction and Preprocessing:** Extracting and cleaning text from PDF files.
  * **Loading Pre-trained Sentence Transformer Model:** Utilizing a lightweight model to produce 384-dimensional embeddings.
  * **Document Segmentation:** Dividing content into logical sections for efficient processing.
  * **Embedding Creation and Cosine Similarity Calculation:** Creating embeddings for document sections and user queries, then using cosine similarity to find relevant sections.
  * **Visualizing Similarity Scores:** Demonstrating how to use `matplotlib.pyplot` to visualize similarity scores.
  * **Implementing Semantic Search Chatbot Function:** Outlining the final step of creating a chatbot that returns relevant sections based on semantic similarity.
  * **Information Retrieval:** Showing how the function identifies information based on meaning rather than just keywords.
  * **RAG-based Application and LLM Integration:** Discussing building RAG applications and integrating Large Language Models (LLMs) for enhanced output control.
  * **Upcoming Topics and API Access:** Mentioning that the next session will cover the GPT API (a paid service), with alternatives like Gemini or Llama to be discussed later.

AI Council also stated that they will share the file in the LMS and upload it.
