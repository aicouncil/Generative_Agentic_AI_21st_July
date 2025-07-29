The document is a summary of a meeting held on July 28, 2025, about Generative & Agentic AI. The AI Council presented on word embedding and a text classification case study.

Here's a breakdown of the key points:

  * **Audio Issues:** The meeting began with GAURAV KUMAR AGARWALLA initiating the session, but the AI Council experienced initial audio issues that were resolved by changing networks and closing their video.
  * **Word Embedding:** The AI Council explained that word embedding converts words into numerical vectors, enabling machine learning to process text data by optimizing how the machine learns from associated weights.
  * **Text Classification Case Study:** A case study was introduced where a model would classify text as "cricket" (labeled 0) or "chess" (labeled 1).
  * **Data Preparation:** The process involved tokenization (breaking text into individual words), padding (uniforming sentence length), and embedding (converting numerical forms into vectors using Keras TensorFlow and NumPy).
  * **Neural Network Model Creation:** A Keras Sequential model was built to convert words into vectors and learn from them. The AI Council detailed using one-hot encoding for unique word vectors.
  * **Embedding Layer Configuration:** The embedding layer required input dimension (vocabulary size), output dimension (embedding dimension), and input length (maximum sentence length).
  * **Hyperparameters vs. Trainable Parameters:** GAURAV KUMAR AGARWALLA asked about output neuron size. The AI Council clarified that this is a "hyperparameter" set by the user, unlike "trainable parameters" (weights) which the machine learns.
  * **Neural Network Architecture and Learning:** Embedded vectors are fed into hidden and output neuron layers. The network learns by adjusting random weights through "backward propagation" to minimize prediction errors.
  * **Model Flattening and Hidden Layers:** The embedding layer's output is flattened into a single input for subsequent layers. Hidden layers are often defined with neurons in multiples of 16.
  * **Model Compilation:** The model was compiled using the Adam optimizer to minimize errors, binary cross-entropy as the loss function for binary classification, and metrics for accuracy.
  * **Training and Troubleshooting:** The model was trained using `model.fit` for 30 epochs. Initial errors were resolved by converting labels to a NumPy array and ensuring consistent sample sizes.
  * **Addressing Low Accuracy:** An initial 50% accuracy was attributed to the linearity of the neural network. A sigmoid activation function was implemented at the output layer, limiting output to 0-1 and improving accuracy to 100%.
  * **Text Prediction and Future Steps:** The model was demonstrated to predict text categories (cricket/chess) by converting new text to numerical sequences and using `model.predict`. Outputs \>0.5 were classified as "chess" and \<0.5 as "cricket." Future topics include language models for word prediction (RNNs) and using pre-built API models like GPT and Gemini.
