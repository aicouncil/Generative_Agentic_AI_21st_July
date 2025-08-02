This document contains notes from a "Generative & Agentic AI" session held on August 1, 2025. The session was led by AI Council, with participants including Ramana Salapu, Prashant Rocks, Souvik Roy, and others.

**Key topics and discussions included:**

  * **Course Content and Expectations:** AI Council emphasized the technical depth of the course, aiming to cover all aspects of GenAI, transformers, BERT, and various LLM models beyond just GPT, despite participants' varying levels of machine learning knowledge.
  * **Freelancing Opportunities:** Ramana Salapu inquired about freelancing for practical experience, and AI Council confirmed the availability of client projects and freelance tasks, emphasizing that time investment in projects leads to long-term benefits.
  * **Importance of Hands-on Experience:** AI Council stressed that continuous learning in data science comes from teaching and working on freelance projects, not just taking courses, to avoid forgetting learned concepts.
  * **Training Continuity and Community Support:** AI Council assured participants that training would continue even with a single attendee and highlighted a strong community for sharing relevant requirements and opportunities.
  * **Hands-on Practice with Hugging Face Transformers Library:** AI Council introduced the Hugging Face `transformers` library for accessing various large language models (LLMs) like LLaMA and GPT-2, demonstrating its use for text and image generation.
  * **Transformer Library Installation and GPU Usage:** Detailed instructions were provided for installing the `transformers` library for both GPU and CPU, with emphasis on checking for Nvidia GPUs and installing TensorFlow as a prerequisite.
  * **Using Google Colab for Accessibility:** Google Colab was recommended as an alternative for users without dedicated GPUs, offering free GPU access and pre-built TensorFlow and `transformers` environments.
  * **BERT for Text Classification:** AI Council explained BERT as an encoder-based transformer used for text classification and sentiment analysis, demonstrating a "hello world" example.
  * **GPT's Decoder-Only Architecture:** The session clarified that GPT models primarily use the decoder part of the transformer architecture, making them suitable for generation tasks, and illustrated how GPT generates text word by word.
  * **Machine Translation with Helsinki NLP Models:** While GPT-2 was shown to be limited for translation, Helsinki NLP models (e.g., `Helsinki-NLP/opus-mt-ja-en`) were introduced as effective encoder-decoder models for translation tasks.
  * **Text Summarization with Google T5 and Facebook's Bart Large CNN:** Google T5 and Facebook's Bart Large CNN models were demonstrated for text summarization, with T5 utilizing both encoder and decoder architectures.
  * **Model Selection Strategies on Hugging Face:** AI Council advised filtering models by task and considering factors like likes and downloads on Hugging Face as indicators of quality and widespread use.
  * **Keyword Extraction Challenges:** Prashant Rocks inquired about keyword extraction, and AI Council explained that dedicated models for this are not directly available on Hugging Face and would require a "prompt-based model" like GPT-3 or Llama.
  * **Llama Model Exploration:** Souvik Roy asked about different Llama models (e.g., Llama 4, Llama 3.1) and their varying parameters and sizes. AI Council explained two methods for accessing Llama models: direct download after approval or via command-line tools like Ollama.
  * **Llama API and Third-Party Access:** It was noted that Llama does not currently offer a direct API but is planning to, and third-party platforms like Groq provide API access to open-source models, including Llama 4, with pricing based on tokens.
  * **Hardware Constraints for LLMs:** Souvik Roy raised concerns about hardware requirements for running LLMs locally, and AI Council emphasized the need for dedicated GPUs and sufficient RAM (16GB or 32GB) for heavy models, suggesting cloud-based environments and APIs as alternatives.

**Suggested next steps from the session include:**

  * AI Council will demonstrate image generation.
  * AI Council will show how to use Llama, Gemini, and DeepSeek as a developer.
  * AI Council will allow the group to use the Llama models.
  * AI Council will explain the process for client investment.
  * AI Council will work on prompt-based models.
