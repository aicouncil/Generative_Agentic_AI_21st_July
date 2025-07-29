# Text Classification with Keras: A Neural Network Approach

This document provides a detailed explanation of building and training a simple neural network for text classification using Keras, as demonstrated in the `gen_g2_d5.py` script. It covers data preparation (integer encoding and padding), model architecture (embedding, flatten, and dense layers), compilation, training, and making predictions on new text.

## 1. Data Preparation

The first step in any machine learning task is to prepare the data in a format suitable for the model. For text data, this involves converting words into numerical representations.

* **Raw Text and Labels:**
    The dataset consists of a list of sentences (`texts`) and corresponding binary `labels` (0 for "cricket" related, 1 for "chess" related).
    ```python
    texts = [
        "I am playing good cricket",
        "He is playing chess",
        "I like to watch cricket",
        "Chess is a mind game",
        "Cricket is played outdoors",
        "Chess pieces are intresting",
        "We played cricket yesterday",
        "He won the chess match"
    ]
    labels = [0 , 1 , 0 , 1 , 0 , 1 , 0 , 1] # 0: cricket, 1: chess
    print(texts)
    print(labels)
    ```

* **Tokenization:**
    The `Tokenizer` from `tensorflow.keras.preprocessing.text` is used to convert text into numerical sequences. It first builds a vocabulary (`word_index`) from the training texts, assigning a unique integer to each unique word.
    ```python
    from tensorflow.keras.preprocessing.text import Tokenizer
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(texts) # Builds vocabulary
    print(tokenizer.word_index)
    # Example Output: {'cricket': 1, 'is': 2, 'chess': 3, 'i': 4, 'playing': 5, ...} (Actual IDs depend on frequency)
    ```
    Then, `texts_to_sequences()` converts the original text sentences into sequences of these integer IDs.
    ```python
    sequences = tokenizer.texts_to_sequences(texts)
    # Example Output: [[4, 5, 6, 1], [7, 2, 5, 3], ...] (IDs vary based on word_index)
    ```

* **Padding Sequences:**
    Since neural networks require inputs of uniform length, `pad_sequences` from `keras.utils` is used. All sequences are padded with zeros to a `max_len` (here, set to 5) either at the beginning (`padding='pre'`) or end.
    ```python
    from keras.utils import pad_sequences
    max_len = 5
    padded_sequences = pad_sequences(sequences , maxlen=max_len, padding='pre')
    print(padded_sequences)
    # Example Output (padded to length 5, zeros at beginning):
    # [[0 0 0 0 1]  (e.g., for "cricket")
    #  [0 0 0 0 1]  (e.g., for "chess")
    #  ...]
    ```
    The `labels` are converted to a NumPy array as required by Keras models.
    ```python
    labels = np.array(labels)
    ```

## 2. Neural Network Model Architecture (Keras Sequential Model)

A sequential model is defined, which is a linear stack of layers, representing the neural network architecture.

* **`models.Sequential()`:** Initializes the neural network model.
* **`layers.Embedding()`:** This is the first layer, crucial for NLP. It converts the integer-encoded words into dense, fixed-size vectors (word embeddings).
    * `input_dim` (`vocab_size`): The size of the vocabulary, which is the number of unique words plus 1 (for padding token 0).
        ```python
        vocab_size = len(tokenizer.word_index) + 1 # Calculates vocabulary size
        # Example Output: vocab_size = 14 (if 13 unique words)
        ```
    * `output_dim` (`embedding_dim=3`): The dimensionality of the dense embedding vectors. Each word will be represented by a vector of 3 numbers.
    * `input_length` (`max_len=5`): The length of the input sequences (after padding).
        ```python
        model.add(layers.Embedding(input_dim = vocab_size, output_dim = embedding_dim, input_length = max_len))
        ```
* **`layers.Flatten()`:** The embedding layer outputs 3D tensors (`(batch_size, input_length, output_dim)`). The `Flatten` layer converts this to a 2D tensor (`(batch_size, input_length * output_dim)`) to be fed into subsequent dense layers.
    ```python
    model.add(layers.Flatten())
    ```
* **`layers.Dense(5)`:** This is a hidden dense (fully connected) layer with 5 neurons. It learns complex patterns from the flattened embeddings.
    ```python
    model.add(layers.Dense(5))
    ```
* **`layers.Dense(1, activation='sigmoid')`:** This is the output layer. For binary classification (predicting 0 or 1), a single neuron (`1`) is used. The `sigmoid` activation function squashes the output to a probability between 0 and 1.
    ```python
    model.add(layers.Dense(1 , activation = 'sigmoid'))
    ```
* **Model Summary:**
    The `model.summary()` provides a detailed breakdown of the model's architecture, including the number of parameters in each layer.
    ```python
    model.summary()
    # Output:
    # Model: "sequential"
    # _________________________________________________________________
    # Layer (type)                Output Shape              Param #
    # =================================================================
    # embedding (Embedding)       (None, 5, 3)              42
    # _________________________________________________________________
    # flatten (Flatten)           (None, 15)                0
    # _________________________________________________________________
    # dense (Dense)               (None, 5)                 80
    # _________________________________________________________________
    # dense_1 (Dense)             (None, 1)                 6
    # =================================================================
    # Total params: 128
    # Trainable params: 128
    # Non-trainable params: 0
    # _________________________________________________________________
    ```
    * *Explanation:* `Param #` for Embedding layer is `vocab_size * embedding_dim` (14 * 3 = 42). For `Dense` layers, it's `(input_neurons * output_neurons) + output_neurons (bias)`.

## 3. Model Compilation and Training

Once the model architecture is defined, it needs to be compiled to configure the learning process, and then trained using the prepared data.

* **Compilation:**
    The `model.compile()` method configures the optimizer, loss function, and metrics.
    * `optimizer = 'adam'`: Adam is a popular optimization algorithm used to efficiently update model weights during training.
    * `loss = 'binary_crossentropy'`: This is the standard loss function for binary classification problems. It quantifies the difference between predicted probabilities and actual binary labels.
    * `metrics = ['accuracy']`: Accuracy is chosen as the metric to monitor during the training process, indicating the proportion of correctly classified samples.
    ```python
    model.compile(optimizer = 'adam', loss = 'binary_crossentropy' , metrics = ['accuracy'])
    ```

* **Training:**
    The `model.fit()` method trains the neural network. It iteratively adjusts the model's weights to minimize the loss function.
    * `padded_sequences`: The input features (the padded integer sequences of the sentences).
    * `labels`: The true labels (0 or 1) for each sentence.
    * `epochs=30`: The number of times the model will pass through the entire dataset. In each epoch, the model makes predictions, calculates the loss, and updates its weights via backpropagation.
    ```python
    model.fit(padded_sequences, labels, epochs=30)
    # Output includes epoch-by-epoch loss and accuracy
    ```

## 4. Making Predictions with the Trained Model

After training, the model can be used to classify new, unseen text.

* **Preprocessing New Text:**
    Any new text must undergo the *same* preprocessing steps (tokenization and padding) as the training data using the *same* `tokenizer` object and `max_len`.
    ```python
    new_text = "Today we have a cricket match"
    new_sequence = tokenizer.texts_to_sequences([new_text])
    pad_seq = pad_sequences(new_sequence , maxlen=5 , padding='pre')
    ```
* **Predicting Probability:**
    The `model.predict()` method outputs a probability (a value between 0 and 1) that the new text belongs to the positive class (class 1, i.e., "chess" in this case).
    ```python
    model.predict(pad_seq) # Output is a probability (e.g., [[0.009...]])
    ```
* **Classifying Based on Threshold:**
    For binary classification, a threshold (commonly 0.5) is applied. If the predicted probability is greater than 0.5, it's classified as class 1; otherwise, it's class 0.
    ```python
    print(int(model.predict(pad_seq) > 0.5))
    # Example Output for "Today we have a cricket match": 0 (correctly classified as cricket)
    ```
* **Another Prediction Example (for "chess"):**
    ```python
    new_text = "Viswanathan Ananad is a legendry chess player"
    new_sequence_chess = tokenizer.texts_to_sequences([new_text])
    pad_seq_chess = pad_sequences(new_sequence_chess , maxlen=5 , padding='pre')
    print(int(model.predict(pad_seq_chess) > 0.5))
    # Example Output: 1 (correctly classified as chess)
    ```

## 5. Use Cases, Benefits, and Limitations

* **Use Cases:**
    * **Sentiment Analysis:** Classifying text as positive, negative, or neutral.
    * **Spam Detection:** Identifying spam emails.
    * **Topic Classification:** Categorizing articles by topic.
    * **Intent Recognition:** Determining a user's intent from their query in chatbots.

* **Benefits of this Approach (Neural Network with Embeddings):**
    * **End-to-End Learning:** The model can learn meaningful representations (embeddings) from the text data itself during training, without requiring manual feature engineering.
    * **Captures Semantic Relationships:** Word embeddings enable the model to understand similarities between words, leading to better generalization.
    * **Scalability:** Can handle large vocabularies and datasets more effectively than traditional sparse methods.

* **Limitations:**
    * **Requires Sufficient Data:** Deep learning models typically need a large amount of labeled data to train effectively and achieve good performance.
    * **Fixed Word Embeddings (for basic Embedding layer):** In this simple setup, each word has a single, fixed embedding regardless of its context. This means words with multiple meanings (polysemy) are not differentiated. More advanced models use contextual embeddings to address this.
    * **Out-of-Vocabulary (OOV) Words:** Words not seen during training will not have an embedding and might be ignored, potentially losing information if they are crucial to the new text's meaning.
    * **Hyperparameter Tuning:** Performance is sensitive to hyperparameters like `embedding_dim`, `max_len`, number of `Dense` layers, neurons, optimizer, and epochs, which often require experimentation.
