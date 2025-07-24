# Word Encodings: Bag of Words (BOW) and TF-IDF

This document provides a detailed explanation of two fundamental text encoding techniques: Bag of Words (BOW) and TF-IDF (Term Frequency-Inverse Document Frequency), as demonstrated in the `gen_g2_d2(word_encodings).py` script. These methods are crucial in Natural Language Processing (NLP) for converting raw text into numerical representations that machine learning algorithms can process.

## 1. Introduction to Word Encodings

In machine learning, models operate on numerical data. Therefore, text data, which is inherently unstructured, must be converted into a structured numerical format. Word encodings are techniques used for this transformation, representing words or documents as vectors of numbers.

## 2. Bag of Words (BOW) Representation

**Definition:** The Bag of Words (BOW) model is a simplified representation used in natural language processing and information retrieval. In this model, a text (such as a sentence or a document) is represented as the bag (multiset) of its words, disregarding grammar and even word order but keeping multiplicity (the number of times a word appears).

* **How it Works:**
    1.  **Tokenization:** The text is first broken down into individual words (tokens).
    2.  **Vocabulary Creation:** A vocabulary of all unique words across the entire set of documents is created.
    3.  **Vector Representation:** Each document is then represented as a numerical vector. The length of this vector is equal to the size of the vocabulary. Each dimension (position) in the vector corresponds to a unique word in the vocabulary. The value in each dimension typically represents the frequency of that word in the document, or its binary presence (1 if the word is in the document, 0 if not), as shown in the provided example.

* **Example from the File (`doc` dataset):**
    Consider the following two documents:
    ```python
    doc = [
        "Rajiv is a good cricket player",
        "Sanjay is bad chess player"
    ]
    print(doc)
    ```

    Using `sklearn.feature_extraction.text.CountVectorizer` with `binary=True`:
    ```python
    from sklearn.feature_extraction.text import CountVectorizer

    vectorizer = CountVectorizer(binary=True)
    vectorizer.fit(doc)
    ```
    The `fit` method learns the vocabulary from the documents. The unique words (features) extracted from the documents are:
    ```
    print(vectorizer.get_feature_names_out())
    # Output: ['bad' 'chess' 'cricket' 'good' 'is' 'player' 'rajiv' 'sanjay']
    ```
    The `transform` method then converts the documents into the binary bag matrix representation:
    ```python
    bag_matrix = vectorizer.transform(doc)
    ```
    This matrix can be visualized as a Pandas DataFrame for better readability:
    ```python
    import pandas as pd
    pd.DataFrame(bag_matrix.toarray(), columns = vectorizer.get_feature_names_out())
    ```
    **Resulting Binary Bag Matrix (DataFrame representation):**

    | | bad | chess | cricket | good | is | player | rajiv | sanjay |
    |---|---|---|---|---|---|---|---|---|
    | 0 | 0 | 0 | 1 | 1 | 1 | 1 | 1 | 0 |
    | 1 | 1 | 1 | 0 | 0 | 1 | 1 | 0 | 1 |

    *Explanation of the matrix*: Each row represents a document, and each column represents a word from the vocabulary. A `1` indicates the presence of the word in that document, and `0` indicates its absence.

* **Use Cases:**
    * **Text Classification:** A common use case where documents are categorized (e.g., spam detection, sentiment analysis). The presence or absence of certain words can be strong indicators for classification.
    * **Document Similarity:** Comparing documents based on their word content.

* **Benefits:**
    * **Simplicity:** Easy to understand and implement.
    * **Effectiveness:** Often performs well for many text classification tasks, especially with sufficient data.

* **Limitations (Explicitly stated in the file):**
    * **No semantic understanding:** BOW models do not capture the meaning or context of words. For example, "not good" is treated similarly to "good" if both words are present.
    * **Ignores word orders and grammar:** The model treats a sentence like a "bag" of words, losing all information about the sequence of words. "Player good cricket" would have the same representation as "good cricket player".
    * **Out of vocabulary (OOV) words are simply ignored:** If the model encounters a word during prediction that was not in its vocabulary built during training, it simply ignores that word. This can lead to loss of information if OOV words are significant.

## 3. TF-IDF Encoding

**Definition:** TF-IDF stands for Term Frequency-Inverse Document Frequency. It is a numerical statistic that is intended to reflect how important a word is to a document in a collection or corpus.

* **How it Works:**
    TF-IDF assigns a weight to each word in a document. This weight increases proportionally to the number of times a word appears in the document (Term Frequency - TF) but is offset by the frequency of the word in the corpus (Inverse Document Frequency - IDF).
    * **Term Frequency (TF):** Measures how frequently a term appears in a document. A higher TF means the word is more relevant to that specific document.
        $TF(t, d) = \text{number of times term t appears in document d}$
    * **Inverse Document Frequency (IDF):** Measures how important a term is across the entire corpus. It downweights common words (like "is", "the", "a") that appear in many documents and upweights rare words that are more distinctive to specific documents.
        $IDF(t, D) = \log\left(\frac{\text{Total number of documents}}{\text{Number of documents with term t}}\right)$
    * **TF-IDF Score:** The TF-IDF score is the product of TF and IDF.
        $TFIDF(t, d, D) = TF(t, d) \times IDF(t, D)$
    Words with high TF-IDF scores are considered highly relevant to a document within the context of the entire collection.

* **Example from the File (`docs` dataset):**
    Consider the following four documents:
    ```python
    docs = [
        "AI is the future",
        "AI and ML are the future",
        "Physics is an intresting subject",
        "I am intrested in the cocepts of physics"
    ]
    ```

    Using `sklearn.feature_extraction.text.TfidfVectorizer`:
    ```python
    from sklearn.feature_extraction.text import TfidfVectorizer

    vectorizer = TfidfVectorizer()
    vectorizer.fit(docs)
    ```
    The `fit` method learns the vocabulary and calculates IDF values. The unique words (features) extracted are:
    ```
    vectorizer.get_feature_names_out()
    # Output: ['ai' 'am' 'an' 'and' 'are' 'cocepts' 'future' 'in' 'intrested' 'is' 'ml' 'of' 'physics' 'subject' 'the']
    ```
    The `transform` method then converts the documents into the TF-IDF weighted matrix:
    ```python
    tf_idf_matrix = vectorizer.transform(docs)
    ```
    This matrix can be visualized as a Pandas DataFrame for better readability:
    ```python
    pd.DataFrame(tf_idf_matrix.toarray(), columns = vectorizer.get_feature_names_out())
    ```
    **Resulting TF-IDF Matrix (DataFrame representation - values are floats representing weights):**

    | | ai | am | an | and | are | cocepts | future | in | intrested | is | ml | of | physics | subject | the |
    |---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
    | 0 | 0.577 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.449 | 0.000 | 0.000 | 0.449 | 0.000 | 0.000 | 0.000 | 0.000 | 0.449 |
    | 1 | 0.457 | 0.000 | 0.000 | 0.355 | 0.355 | 0.000 | 0.355 | 0.000 | 0.000 | 0.000 | 0.457 | 0.000 | 0.000 | 0.000 | 0.355 |
    | 2 | 0.000 | 0.000 | 0.428 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.334 | 0.334 | 0.000 | 0.000 | 0.428 | 0.428 | 0.000 |
    | 3 | 0.000 | 0.370 | 0.000 | 0.000 | 0.000 | 0.370 | 0.000 | 0.370 | 0.288 | 0.000 | 0.000 | 0.370 | 0.288 | 0.000 | 0.000 |

    *Explanation of the matrix*: Each cell contains a floating-point number representing the TF-IDF weight of a specific word in a specific document. Higher values indicate greater importance of the word to that document relative to the corpus. Notice how common words like "is" and "the" have lower weights compared to more specific terms like "AI" or "physics".

* **Use Cases:**
    * **Information Retrieval:** Ranking documents by relevance to a user query.
    * **Text Summarization:** Identifying keywords or sentences that best represent the main topic of a document.
    * **Keyword Extraction:** Automatically extracting the most important terms from a document.

* **Benefits:**
    * **Considers Word Importance:** Addresses a limitation of BOW by assigning weights that reflect a word's relevance, reducing the impact of very common words.
    * **Better for Ranking:** Provides a more nuanced understanding of document content for tasks like search and document similarity.

* **Limitations (Explicitly stated in the file):**
    * **No semantic understanding:** Like BOW, TF-IDF does not capture the meaning or context of words. It treats "apple" (fruit) and "Apple" (company) as the same token if not preprocessed, and doesn't understand synonyms or paraphrasing.
    * **Ignores word orders and grammar:** It still relies on the bag-of-words assumption, meaning the sequence of words is lost. "Dog bites man" and "Man bites dog" would have the same TF-IDF vector.

## 4. Shared Limitations of BOW and TF-IDF

Both Bag of Words and TF-IDF are "count-based" vectorization techniques that share fundamental limitations due to their simplified approach to text representation:
* They lack **semantic understanding**, meaning they cannot grasp the meaning, context, or relationships between words.
* They **ignore word order and grammar**, treating text as an unordered collection of words. This means the nuanced meaning conveyed by word sequence is lost.
* They struggle with **out-of-vocabulary (OOV) words**, which are words not seen during training. These words are typically ignored during transformation, leading to information loss.

These limitations highlight the need for more advanced text embedding techniques in modern NLP, such as word embeddings (e.g., Word2Vec, GloVe) and contextual embeddings (e.g., BERT, GPT), which aim to capture semantic and contextual information.
