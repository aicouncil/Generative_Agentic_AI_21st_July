The provided file is a summary of a meeting titled "Generative & Agentic AI - 22 July" that took place on July 24, 2025. The notes were taken by Gemini.

Here's a summary of the key topics discussed:

  * **Word Encoding and Numerical Representation:** Traditional methods like Bag of Words and TF-IDF don't preserve word sequence. Integer encoding assigns unique integer values to words, preserving some sequential information, which is useful for RNNs and Transformers.
  * **Implementing Integer Encoding with TensorFlow:** TensorFlow is recommended for large-scale integer encoding due to its efficiency.
  * **Keras and Tokenization:** Keras, built on TensorFlow, simplifies neural network creation. The `Tokenizer` class in Keras is used for tokenizing text and converting it to numerical sequences.
  * **Padding for Structured Data:** Padding (using `pad_sequences`) is used to make all numerical sequences uniform in length, structuring the data for consistent processing.
  * **Machine Learning Model Learning Process:** Models learn by finding relationships between inputs and outputs using trainable parameters (weights), iteratively adjusting them to minimize error.
  * **Model Size and Trainable Parameters:** The number of trainable parameters determines model size. Large datasets require more parameters to avoid "underfit" models.
  * **Neural Networks for Increased Trainable Parameters:** Neural networks, with hidden layers and neurons, increase trainable parameters, allowing models to learn complex patterns in large datasets.
  * **Challenges with Integer Encoding for Large Data:** Directly using integer encoding with thousands of unique words can create very large vectors and introduce artificial bias due to sequential integer assignment.
  * **Introduction to Word Embedding and One-Hot Encoding:** Word embedding overcomes integer encoding limitations for large datasets. It starts with one-hot encoding, representing each word as a vector with a single '1'.
  * **Word Embedding and Vectorization:** Word embedding converts words into vectors, enabling models to learn more trainable parameters and weights for each word.
  * **Practical Implementation of Word Embedding:** This involves one-hot encoding using `keras.utils.to_categorical` and an embedding layer to convert words into their vector representations.
  * **Importance of Vectorization in AI:** Conceptual understanding of "word vector" and "vector database" is crucial for generative AI and RAG-based applications, as models rely on structured, vectorized data.
  * **Addressing Programming Concerns:** Hands-on practice is important for programming, but the main goal of the discussion was to convey the concept of word vectorization in generative AI. Padding helps convert unstructured data into structured data.

