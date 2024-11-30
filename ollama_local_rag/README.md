# Instructions

### RAG Setup
Running `create_embeddings.py` will create the embeddings for the data in `knowledge_base/` and store them in `chroma_db/`. Run `create_embeddings.py --rerun` if you modify your knowledge base to regen the embeddings. Then you can run `main.py` to run an interactive prompt with ollama and you can run with `-d [QUESTION]` if you want a straight answer.

### Poisoning
Inside `knowledge_base/malicious`, insert the text files that you want to poison with (I've already provided a sample one messing with Lugia's Pokedex number)

### Testing Success Rate
Run `create_embeddings.py --rerun` to recreate embeddings with your new malicious text, then modify the top portions of `attack_evaluation.py` and then run it. It will create some statistics on poisoning success rate.

## Installation
Go [here](https://ollama.com/download/windows) and download ollama and run llama3.2 with it.

Install the `requirements.txt`