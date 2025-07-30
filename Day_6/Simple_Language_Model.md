# Building a Simple Language Model with Keras (RNN)

This document provides a detailed explanation of building a basic language model using a Recurrent Neural Network (RNN) in Keras, focusing on text preprocessing, sequence generation, and model training. The content is derived from the `gen_g2_d6.py` script.

## 1. Data Loading and Initial Inspection

The script begins by loading text data from a local file, simulating a corpus for language modeling.

* **File Reading:** A text file named `/content/IndiaUS.txt` is opened in read mode (`'r'`), its entire content is read into the `mytext` variable, and then the file is closed.
    ```python
    myfile = open('/content/IndiaUS.txt' , 'r')
    mytext = myfile.read()
    myfile.close()
    ```
* **Text Content:** The raw text `mytext` is printed to show the loaded data.
    ```python
    print(mytext) # Output: The actual content of IndiaUS.txt
    ```

## 2. Text Preprocessing: Tokenization and Vocabulary Creation

Before building a language model, the raw text needs to be processed into a numerical format.

* **Tokenization:**
    * `tensorflow.keras.preprocessing.text.Tokenizer` is used to create a tokenizer object.
    * `mytokenizer.fit_on_texts([mytext])` builds a vocabulary by assigning a unique integer ID to each unique word in the text. This process is case-insensitive and typically removes punctuation.
    ```python
    from tensorflow.keras.preprocessing.text import Tokenizer
    mytokenizer = Tokenizer()
    mytokenizer.fit_on_texts([mytext])
    ```
* **Vocabulary (Word Index):** The `word_index` attribute of the tokenizer stores the mapping from words to their integer IDs.
    ```python
    word_index = mytokenizer.word_index
    print(word_index) # Output: A dictionary like {'the': 1, 'india': 2, 'and': 3, ...}
    ```
* **Vocabulary Size:** The total number of unique words in the vocabulary (including an implicit ID 0 for padding/OOV, though not explicitly shown to be used for OOV in this script) is calculated as `len(word_index) + 1`.
    ```python
    total_words = len(word_index) + 1
    print("Vocabulary Size- " , total_words) # Output: Vocabulary Size- 599 (for the given text)
    ```

## 3. N-gram Sequence Generation for Language Modeling

Language models learn to predict the next word in a sequence. To train such a model, the text is converted into sequences of n-grams (subsequences of n words) where the first `n-1` words are the input and the `n`-th word is the target.

* **Splitting Text into Lines:** The `mytext` is split into individual lines based on the newline character (`\n`).
    ```python
    lines = mytext.split("\n")
    print(lines[:3]) # Output: First 3 lines of the text
    ```
* **Generating Input Sequences (N-grams):**
    The script iterates through each line and then through the tokens within each line to create various n-gram sequences. For a tokenized list like `[A, B, C, D]`, it generates:
    * `[A, B]` (input: `A`, target: `B`)
    * `[A, B, C]` (input: `A, B`, target: `C`)
    * `[A, B, C, D]` (input: `A, B, C`, target: `D`)
    Each of these n-gram sequences is appended to `my_input_sequences`.
    ```python
    my_input_sequences = []
    for line in lines:
      token_list = mytokenizer.texts_to_sequences([line])[0] # Convert line to sequence of IDs
      for i in range(1,len(token_list)):
        n_gram_sequences = token_list[0:i+1] # Create n-gram
        my_input_sequences.append(n_gram_sequences)
    print(my_input_sequences) # Output: List of all generated n-gram sequences
    ```

## 4. Sequence Padding

Neural networks require input sequences of uniform length. Padding is used to achieve this uniformity.

* **Determine Max Sequence Length:** The maximum length among all generated n-gram sequences is found. This length will be used for padding.
    ```python
    sequence_lengths = []
    for sequence in my_input_sequences:
      sequence_lengths.append(len(sequence))
    print(max(sequence_lengths)) # Output: 83
    ```
* **Padding:** `tensorflow.keras.preprocessing.sequence.pad_sequences` is used to pad all sequences in `my_input_sequences` to the `maxlen` (here, 83) by adding zeros at the beginning (`padding='pre'`).
    ```python
    from tensorflow.keras.preprocessing.sequence import pad_sequences

    input_sequences = pad_sequences(my_input_sequences , maxlen=83, padding='pre')
    print(input_sequences) # Output: Padded 2D NumPy array
    print(type(input_sequences)) # Output: <class 'numpy.ndarray'>
    ```

## 5. Splitting Features (X) and Labels (y)

For training a language model, the padded sequences need to be split into input features (`X`) and corresponding target labels (`y`).

* **Features (X):** Consists of all words in each padded sequence *except* the last one. This is the context provided to the model.
    ```python
    X = input_sequences[ : , :-1 ] # All columns except the last one
    print(X.shape) # Output: (number_of_sequences, 82)
    ```
* **Labels (y):** Consists of only the last word in each padded sequence. This is the word the model is supposed to predict given the preceding context.
    ```python
    y = input_sequences[ : , -1: ] # Only the last column
    print(y.shape) # Output: (number_of_sequences, 1)
    ```
* **One-Hot Encode Labels:** The target labels (`y`) are then one-hot encoded using `keras.utils.to_categorical`. This is necessary for a multi-class classification problem where the model predicts probabilities for each possible next word in the vocabulary.
    * The `num_classes` for `to_categorical` implicitly becomes `total_words` (599).
    ```python
    from keras.utils import to_categorical
    Yt = to_categorical(y)
    print(Yt[0]) # Output: One-hot encoded vector for the first label
    print(Yt.shape) # Output: (number_of_sequences, total_words) (e.g., (1166, 599))
    ```

## 6. Neural Network Model Architecture (RNN)

A sequential Keras model is defined using an `Embedding` layer followed by a `SimpleRNN` layer and a `Dense` output layer.

* **`models.Sequential()`:** Initializes the neural network model.
* **`layers.Embedding()`:**
    * `input_dim = total_words`: The size of the vocabulary (599 unique words + 1 for padding).
    * `output_dim = 16`: Each word will be converted into a dense vector of 16 dimensions.
    * `input_length = 82`: The length of each input sequence (`X`), which is `max_len - 1`.
    ```python
    from tensorflow.keras import models, layers
    model = models.Sequential()
    model.add(layers.Embedding(input_dim = total_words, output_dim = 16, input_length = 82))
    ```
* **`layers.SimpleRNN()`:** A Simple Recurrent Neural Network layer. RNNs are designed to process sequential data, making them suitable for language modeling. The `82` within `SimpleRNN(82)` refers to the number of recurrent units (neurons) in this layer. This layer processes the embedded sequences and maintains an internal "memory" of previous tokens.
    ```python
    model.add(layers.SimpleRNN(82))
    ```
* **`layers.Dense()` (Output Layer):** This is a fully connected output layer.
    * `599` neurons: Corresponds to `total_words`, meaning it will output a probability distribution over all words in the vocabulary for the next word prediction.
    * `activation = 'sigmoid'`: The sigmoid activation function is used here, suitable for binary classification or when treating each output neuron independently as a probability. For multi-class probability distributions, `softmax` is more common, but `sigmoid` can be used if outputs are interpreted as independent probabilities or if `total_words` is not the actual number of classes. It might be a simplification or a specific design choice in this example.
    ```python
    model.add(layers.Dense(599 , activation = 'sigmoid'))
    ```
* **Model Summary:** Provides an overview of the model's layers and trainable parameters.
    ```python
    model.summary()
    # Output:
    # Model: "sequential"
    # _________________________________________________________________
    # Layer (type)                Output Shape              Param #
    # =================================================================
    # embedding (Embedding)       (None, 82, 16)            9584  (599 * 16)
    # _________________________________________________________________
    # simple_rnn (SimpleRNN)      (None, 82)                8112  (82*16 + 82*82 + 82)
    # _________________________________________________________________
    # dense (Dense)               (None, 599)               49511  (82*599 + 599)
    # =================================================================
    # Total params: 67207
    # Trainable params: 67207
    # Non-trainable params: 0
    # _________________________________________________________________
    ```

## 7. Model Compilation and Training

The model is compiled to define its optimizer, loss function, and metrics, and then trained using the prepared data.

* **Compilation:**
    * `optimizer = 'adam'`: An efficient adaptive learning rate optimizer.
    * `loss = 'categorical_crossentropy'`: This is the standard loss function for multi-class classification problems where labels are one-hot encoded.
    * `metrics = ['accuracy']`: Monitors the prediction accuracy during training.
    ```python
    model.compile(optimizer = 'adam', loss = 'categorical_crossentropy' , metrics = ['accuracy'])
    ```
* **Training:**
    The `model.fit()` method trains the model for a specified number of `epochs` (100 in this case). The model learns to map input sequences (`X`) to their corresponding one-hot encoded next-word labels (`Yt`).
    ```python
    model.fit(X,Yt,epochs=100)
    # Output shows epoch-by-epoch loss and accuracy, indicating learning progress.
    ```

## 8. Use Cases, Benefits, and Limitations

* **Use Cases (Language Models):**
    * **Text Generation:** Generating coherent and contextually relevant new text.
    * **Next Word Prediction:** Auto-completion in search engines or typing suggestions.
    * **Machine Translation:** Translating text from one language to another.
    * **Speech Recognition:** Converting spoken words into text.

* **Benefits of this Approach (RNN-based Language Model):**
    * **Sequential Data Processing:** RNNs are inherently designed to handle sequences, allowing them to capture dependencies and context over time (or across words in a sentence).
    * **Learned Word Representations:** The `Embedding` layer learns dense, meaningful vector representations of words.
    * **Predictive Power:** Can learn complex patterns in text to predict subsequent elements.

* **Limitations (Specific to this Simple RNN Language Model):**
    * **Vanishing/Exploding Gradients:** Simple RNNs can struggle with learning long-range dependencies due to vanishing or exploding gradients. More advanced RNN architectures (LSTMs, GRUs) mitigate this.
    * **Limited Context Window:** While RNNs have "memory," their effective memory for very long sequences can be limited.
    * **Computational Cost:** Training RNNs can be computationally intensive, especially for large vocabularies and long sequences.
    * **Output Layer Activation:** Using `sigmoid` activation for a multi-class output (599 neurons) is unconventional; `softmax` is typically used to produce a probability distribution that sums to 1 across all classes. This might be a simplification in the example or intended for independent binary classification on each word's presence.
