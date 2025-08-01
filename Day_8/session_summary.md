This document is a summary of the "Generative & Agentic AI - 22 July" meeting from July 31, 2025.

**Key topics covered by AI Council:**

  * **Introduction to Transformers:** Explained the architecture of transformers, including encoders and decoders, multi-head self-attention, normalization, and feed-forward networks. The process begins with input data undergoing embedding and positional encoding before multi-head self-attention.
  * **Self-Attention Mechanism:** Detailed how self-attention allows each word in a sentence to focus on other words relevant to its meaning, using examples like "I buy a red car" and "the cat sat on the mat because it was time" to illustrate how models understand relationships between words (e.g., "it" referring to "cat").
  * **QKV Vectors in Self-Attention:** Introduced Query (Q), Key (K), and Value (V) vectors, explaining their roles: Q ("what am I looking for"), K ("what do I offer"), and V ("what do I say if you pay attention to me"). These vectors help identify important word relationships.
  * **Attention Score Calculation:** Explained that attention scores are calculated by the dot product of a Query (Q) vector and a Key (K) vector, indicating how much focus one word should give to another. The core formula for attention, Softmax(Q multiplied by KT / square root of D\_K) multiplied by V, was presented.
  * **Word Embedding and QKV Production:** Demonstrated how words are initially represented as numerical vectors (word embedding) and then transformed into Q, K, and V vectors through linear neural layers, which are learned during training.
  * **Scaling and Softmax for Stability:** Emphasized the importance of scaling by dividing by the square root of the dimension of K (DK) to prevent large or small attention scores from causing instability. Softmax is then applied to normalize these scaled values into probabilities.
  * **Weighted Sum of Value Vectors for Contextual Output:** Described how attention weights are used to compute a weighted sum of Value (V) vectors, resulting in a "context-aware" output vector that captures the contextual understanding of a word within a sentence. This allows models to differentiate between multiple meanings of a word (e.g., "Apple" as a fruit vs. a company).
  * **Practicality and Future Implications of Attention Mechanism:** Stressed that while practitioners don't directly use this attention process for building applications, understanding it is crucial for grasping future advancements in AI, such as AI agents, and adapting to evolving technology.
  * **Contextual Understanding in Language Models:** Explained how transformers, unlike RNNs, can understand word meanings even when words are far apart in a sentence due to parallel processing, leading to faster operations.
  * **Multi-Head Attention:** Briefly introduced multi-head attention as a mechanism that allows models to learn relationships based on various factors like grammar and word positioning, stating it would be explained in detail in an upcoming class.

**Upcoming Sessions and Hands-on Activities:**

  * AI Council will explain multi-head attention and the encoder-decoder in detail in the next class.
  * The next session will include hands-on activities using GPT and other tools.
  * AI Council will provide instructions to Souvik Roy on how to install and use the smallest LLaMA model (1.8GB) on their system, noting it requires at least 2.5 GB of storage space and 8 GB of RAM.
