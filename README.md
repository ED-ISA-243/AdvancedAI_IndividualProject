AI Recipe Gnerator in this repository and guide how to setup in this README.

Note: the GPT-NEO-1.3B.ipynb file is an attemt to make the application with another offline AI, but this approach was let go and replaced by Ollama. The file is still in this directory to show my additional work, but should not be initialized and run. If you want to do this anyways, here is how:

1. pip install torch transformers in a terminal

2. write this code in a py file and run it, this will download the model. Then you can run the ipynb file:

"" from transformers import GPTNeoForCausalLM, GPT2Tokenizer

    model_name = "EleutherAI/gpt-neo-1.3B"
    GPTNeoForCausalLM.from_pretrained(model_name)
    GPT2Tokenizer.from_pretrained(model_name) ""




Here are the steps in order to run ollama_ai_app.py:


Step 1 (recommended):
    python3 -m venv venv
    source .venv/bin/activate

Step 2, install dependencies:
    pip install streamlit ollama

Step 3, download Ollama from its official website:
    https://ollama.com/download

Step 4, install the model used in this project in the venv:
    ollama run llama3:latest
    (let it fully download and initialize)

Step 5, run this command into the terminal:
    streamlit run path/to/your/ollama_ai_app.py

Note: you can also test if the model works on your laptop by entering "ollama run llama3.2" in your terminal. there you can also talk to the AI. This following but of text is an example of interaction with the AI.



ollama run llama3.2
>>> hey
How's it going? Is there something I can help you with or would you like to chat?

>>> what model are you?
I am a transformer-based language model, specifically a variant of the BERT (Bidirectional Encoder Representations from Transformers) architecture. My developers use a 
combination of techniques such as self-supervised learning and fine-tuning on large datasets to generate my responses.

More specifically, I'm based on the Hugging Face Transformers library, which is an open-source framework for building and training natural language processing models. 
My model is trained on a massive dataset of text from the internet, books, and other sources, which allows me to understand and generate human-like language.

However, it's worth noting that I'm not a single, monolithic model - I'm actually a collection of smaller models that work together to generate responses. This allows 
me to handle a wide range of tasks and domains with varying degrees of accuracy.

>>> /bye