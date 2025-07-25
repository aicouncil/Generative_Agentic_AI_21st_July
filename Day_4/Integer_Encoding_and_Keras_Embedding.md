# Text Data Preprocessing: Integer Encoding and Keras Embedding

This document provides a detailed explanation of two fundamental steps in preparing text data for machine learning models: Integer Encoding and Word Embedding using Keras, as demonstrated in the `gen_g2_d2(word_encodings).py` script. These techniques are crucial for converting human-readable text into a numerical format that deep learning models can understand and process.

## 1. Integer Encoding (Tokenization and Sequencing)

**Definition:** Integer encoding, also known as tokenization and sequencing, is a process of converting unique words in a text corpus into unique numerical identifiers (integers). Each distinct word is assigned a specific integer.

* **Process and How it Works:**
    The `tensorflow.keras.preprocessing.text.Tokenizer` class is used to perform integer encoding.

    1.  **Tokenizer Initialization:** An instance of `Tokenizer` is created.
        ```python
        from tensorflow.keras.preprocessing.text import Tokenizer

        tokenizer = Tokenizer()
        ```

    2.  **Fitting on Texts (Vocabulary Creation):** The `fit_on_texts()` method processes the input `texts` to build an internal vocabulary. It creates a `word_index` dictionary where each unique word is mapped to a unique integer ID. The words are assigned IDs based on their frequency (most frequent gets ID 1, second most frequent gets ID 2, and so on).
        ```python
        texts = ["Generative AI is intresting",
                 "AI is transforming the world",
                 "I want to know about AI more."]

        tokenizer.fit_on_texts(texts)
        ```
        **Example of `word_index` (Vocabulary):**
        ```python
        print(tokenizer.word_index)
        # Output: {'ai': 1, 'is': 2, 'the': 3, 'generative': 4, 'intresting': 5, 'transforming': 6, 'world': 7, 'i': 8, 'want': 9, 'to': 10, 'know': 11, 'about': 12, 'more': 13}
        ```
        *Note: The IDs start from 1. ID 0 is typically reserved for padding or out-of-vocabulary (OOV) tokens.*

    3.  **Converting Texts to Sequences:** The `texts_to_sequences()` method converts the input texts into lists of integers, where each integer corresponds to the `word_index` of the word.
        ```python
        sequences = tokenizer.texts_to_sequences(texts)
        # Output: [[4, 1, 2, 5], [1, 2, 6, 3, 7], [8, 9, 10, 11, 12, 1, 13]]
        ```
        *Explanation:* For "Generative AI is intresting", 'Generative' (ID 4), 'AI' (ID 1), 'is' (ID 2), 'intresting' (ID 5) become `[4, 1, 2, 5]`.

    4.  **Padding Sequences:** Neural networks typically require input sequences of uniform length. `keras.utils.pad_sequences` is used to achieve this by adding zeros to the beginning (`padding='pre'`) or end of sequences until they all reach the length of the longest sequence in the batch.
        ```python
        from keras.utils import pad_sequences
        padded_sequences = pad_sequences(sequences , padding = 'pre')
        # Output:
        # [[ 0  0  0  4  1  2  5]
        #  [ 0  0  1  2  6  3  7]
        #  [ 8  9 10 11 12  1 13]]
        ```
        *Explanation:* The longest sequence is 7 elements long. Shorter sequences are padded with zeros at the beginning to match this length.

    5.  **One-Hot Encoding (Mentioned as an alternative for sparse representation):** The script also shows `to_categorical`, which would convert integer sequences into a one-hot encoded format. While `to_categorical` can be part of text processing, it's typically used when the labels are categorical, not directly for word embedding as a dense representation. Here, it is shown as a separate conversion, not necessarily as a step *before* embeddings.
        ```python
        from keras.utils import to_categorical
        to_categorical(padded_sequences)
        ```
        *Output (truncated for brevity):* This would result in a very large, sparse matrix where each row corresponds to a word in the padded sequence, and columns represent vocabulary words, with a '1' at the position of the word's ID and '0's elsewhere.

* **Use Cases:**
    * **Preprocessing:** Integer encoding is a mandatory first step before feeding text data into neural network embedding layers or other NLP models that require numerical inputs.
    * **Vocabulary Management:** The `word_index` provides a clear mapping and allows for managing the size of the vocabulary.

* **Benefits:**
    * **Converts Text to Numbers:** Essential for machine learning models.
    * **Unique Representation:** Each word gets a unique ID.
    * **Handles Varying Lengths:** Padding ensures uniform input size for neural networks.

* **Limitations:**
    * **Arbitrary Relationship:** The integer IDs themselves have no inherent meaning or relationship. For example, word ID 1 and word ID 2 are not semantically closer than word ID 1 and word ID 100.
    * **High Dimensionality (if used as sparse input):** If used directly as a sparse one-hot encoded input without a subsequent embedding layer, it can lead to very high-dimensional vectors for large vocabularies.
    * **Out-of-Vocabulary (OOV) Words:** Words not present in the training vocabulary will not have an assigned integer ID, often leading to them being ignored or mapped to a special OOV token (which is not explicitly done here but is a common practice).

## 2. Keras Embedding Layer (Word Embeddings)

**Definition:** Word embeddings are dense vector representations of words. Unlike sparse representations (like one-hot encoding), word embeddings capture semantic relationships between words, meaning words with similar meanings or contexts will have similar vector representations in a multi-dimensional space. This allows models to generalize better and requires fewer dimensions than one-hot encoding for large vocabularies.

* **Process and How it Works:**
    A Keras `Embedding` layer is typically the first layer in a neural network when working with integer-encoded text data. It learns dense vector representations for each word in the vocabulary.

    1.  **Sequential Model:** A simple sequential model is created, which is a linear stack of layers.
        ```python
        from tensorflow.keras import models, layers

        model = models.Sequential()
        ```

    2.  **Embedding Layer Definition:** An `Embedding` layer is added as the first layer.
        ```python
        model.add(layers.Embedding(input_dim = 14 , output_dim = 4 , input_length = 7))
        ```
        * **`input_dim` (Vocabulary Size):** This is the size of the vocabulary, representing the total number of unique words plus one (for ID 0, used for padding or OOV). In the given example, `tokenizer.word_index` has 13 unique words, so `input_dim` is set to 14.
        * **`output_dim` (Embedding Dimension):** This specifies the size of the dense vector for each word. A smaller dimension (e.g., 4 in this case) is chosen for simplicity, but in real-world scenarios, it can range from 50 to 300 or more, depending on the complexity of the task and vocabulary size.
        * **`input_length` (Sequence Length):** This is the length of the input sequences that will be fed to the embedding layer. It should match the length of the padded sequences (7 in this case).

    3.  **Model Summary:** The `model.summary()` method provides a concise overview of the model's layers, output shapes, and the number of trainable parameters.
        ```python
        model.summary()
        # Output:
        # Model: "sequential"
        # _________________________________________________________________
        #  Layer (type)                Output Shape              Param #
        # =================================================================
        #  embedding (Embedding)       (None, 7, 4)              56
        # =================================================================
        # Total params: 56
        # Trainable params: 56
        # Non-trainable params: 0
        # _________________________________________________________________
        ```
        *Explanation of `Param # (56)`:* This is calculated as `input_dim * output_dim` (14 unique words * 4 dimensions per word = 56 parameters). These are the weights that the embedding layer will learn during model training to represent each word.

    4.  **Generating Word Vectors (Prediction):**
        The model is compiled (though the optimizer `'adam'` is specified, it's not actually trained in this snippet, only `predict` is called). The `predict()` method is then used to pass the `padded_sequences` through the embedding layer, which returns the initial (randomly initialized) or learned (if trained) word vectors for each word in each sequence.
        ```python
        model.compile('adam') # Compiles the model
        word_vectors = model.predict(padded_sequences) # Generates word vectors
        # Output (example, values will vary due to random initialization):
        # [[[0.00311211 -0.01538355  0.02700344  0.01188371]
        #   [0.00311211 -0.01538355  0.02700344  0.01188371]
        #   [0.00311211 -0.01538355  0.02700344  0.01188371]
        #   [0.02875153 -0.04351333 -0.02741162  0.04411136]
        #   ... (truncated)
        # ]]]
        ```
        *Explanation:* `word_vectors` is a 3D NumPy array: `(batch_size, sequence_length, embedding_dimension)`. For example, `word_vectors[0]` represents the embedded sequence of the first sentence, `word_vectors[0][0]` would be the vector for the first word in that sentence, and `word_vectors[0][0][0]` would be the first dimension of that word's vector.

    5.  **Accessing Sentence Word Vectors:** The script shows how to access and flatten the word vectors for individual sentences.
        ```python
        #Sentence 1 - Generative AI is intresting
        word_vectors[0].flatten() # Flattens the 7x4 matrix into a 1D array of 28 elements
        #Sentence 2 - AI is transforming the world
        word_vectors[1].flatten()
        #Sentence 3 - I want to know about AI more.
        word_vectors[2].flatten()
        ```
        *Explanation:* `flatten()` converts the 2D array representing a sentence's embedded words (e.g., 7 words * 4 dimensions) into a single 1D array, which can be useful for downstream tasks or further processing, though typically the 3D output is fed into subsequent layers like LSTMs or GRUs.

* **Use Cases:**
    * **Input Layer for Neural Networks:** Word embeddings form the initial input layer for most deep learning models in NLP tasks such as sentiment analysis, machine translation, text summarization, and question answering.
    * **Semantic Similarity:** Learned embeddings can be used to calculate semantic similarity between words (e.g., "king" and "queen" would be closer in the embedding space than "king" and "table").

* **Benefits:**
    * **Captures Semantic Relationships:** Words with similar meanings or contexts are mapped to similar vector spaces, allowing the model to generalize patterns learned from one word to semantically related words.
    * **Dimensionality Reduction:** Converts high-dimensional and sparse one-hot encoded vectors into lower-dimensional, dense representations, making computation more efficient.
    * **Efficient for Large Vocabularies:** Scales much better than one-hot encoding for large vocabularies.

* **Limitations:**
    * **Context-Independent:** Basic Keras `Embedding` layers assign a single, fixed embedding vector to each word, regardless of its context. This means polysemous words (words with multiple meanings, like "bank") will have the same representation in all contexts.
    * **Random Initialization:** The embeddings are typically randomly initialized and need to be learned during the training of the neural network. For smaller datasets, pre-trained embeddings (e.g., Word2Vec, GloVe) might be preferred.
    * **OOV Words:** If an OOV word is encountered during inference and not handled during tokenization, it will not have a valid embedding.

## 3. Relationship Between Integer Encoding and Keras Embedding

Integer encoding is a prerequisite for using a Keras `Embedding` layer. The `Embedding` layer takes integer-encoded sequences as input. These integers serve as indices to look up the corresponding word vectors (embeddings) in an internal embedding matrix. The `input_dim` of the `Embedding` layer is directly determined by the size of the vocabulary generated by the integer encoding process.
