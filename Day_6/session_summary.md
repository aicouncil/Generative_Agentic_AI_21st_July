Here's a summary of the meeting notes from "Generative & Agentic AI - 22 July - 2025/07/29 20:59 GMT+05:30":

The AI Council discussed generative AI in NLP, specifically the limitations of deep neural networks for text generation and introduced Recurrent Neural Networks (RNNs) as a solution. They explained four RNN architectures (one-to-one, one-to-many, many-to-one, and many-to-many) and focused on the many-to-one architecture for predicting the next word.

The implementation steps for an RNN model were demonstrated, including:

  * **Reading and Processing Text Data:** Downloading, uploading to Google Colab, and loading raw text using Python's file handling.
  * **Tokenization and Word Indexing:** Using `tensorflow.keras.preprocessing.text.Tokenizer` to create a word index, assigning unique numerical values to each word, and accounting for padding (zero).
  * **Preparing Sequential Data:** Splitting text into individual lines and converting them into numerical sequences using `text_to_sequences` method.
  * **Creating Input-Target Pairs:** Generating overlapping `n-gram` sequences (e.g., "99" as input, "4" as target; "99, 4" as input, "177" as target) to prepare data for training.
  * **Data Preparation and Padding:** Identifying the maximum sequence length (83) and using `pad_sequences` to ensure all sequences are of uniform length.
  * **Feature and Target Column Definition:** Separating the last column as the target (Y) and the rest as features (X) using NumPy slicing.
  * **Model Building and Initial Training:** Constructing a sequential model with an embedding layer (output dimension 16), a Simple RNN layer (82 units), and a dense output layer. The model was compiled and fitted with 100 epochs, but initially showed very low accuracy (5-6%).
  * **Addressing Low Accuracy with One-Hot Encoding:** The low accuracy was attributed to non-binary target values. To resolve this, the target column was converted to a one-hot encoded format using Keras's `to_categorical` utility. This allowed the model to predict across all 599 possible words by creating a corresponding number of output neurons.
  * **Model Improvement:** After adjusting the model to use categorical cross-entropy as the loss function, the accuracy significantly improved to 98%.

**Next steps for the AI Council include:**

  * Predicting the next word and re-explaining the concepts.
  * Sharing the file in the classroom and meeting.
