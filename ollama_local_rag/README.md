# Instructions
Running `create_embeddings.py` will create the embeddings for the data in `knowledge_base/` and store them in `chroma_db/`. Run `create_embeddings.py --rerun` if you modify your knowledge base to regen the embeddings. Then you can run `main.py` to run an interactive prompt with ollama and you can run with `-d [QUESTION]` if you want a straight answer.

## Installation
Go [here](https://ollama.com/download/windows) and download ollama and run llama3.2 with it.

Install the `requirements.txt`