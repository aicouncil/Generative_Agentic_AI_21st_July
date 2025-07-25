# -*- coding: utf-8 -*-
"""gen_g2_d1(transformers).ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1Qewg6-Qy06e-Ski7RLFkAhRNX-zyFm-2
"""

!pip install transformers

from transformers import pipeline

generator = pipeline('text-generation', model='gpt2-large')

#Generate text
prompt = "In the future, artificial intelligence will"
output = generator(prompt,
                   max_length=50,
                   num_return_sequences=3,
                   top_k = 20,
                   top_p = 0.95,
                   temperature = 0.7,
                   eos_token_id = 50256
                   )

"""* **tok_k = 20 - Considers only top 20 most probable next words.**
* **tok_p = 0.95 - Uses nucleus sampling to include words whose cumulative probability is 95%**
* **temperature = 0.9 - Adjust randomness in generation**
    * temperature = 1.0 => Normal randomness
    * temperature < 1.0 => Less random , more focused,.
    * temperature > 1.0 => More random, more surprising (sometimes output may be nonsense)
* **eos_token_id = 50256 - Special token indicating the end of a sequence in GPT-2**
"""

print(output[0]['generated_text'])

print(output[1]['generated_text'])

