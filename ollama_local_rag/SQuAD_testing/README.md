# Instructions

### RAG Setup
Running `create_embeddings.py` will create the embeddings for the data pulled from SQuAD dataset and store them in `chroma_db/`. Then you can run `main.py` to run an interactive prompt with ollama and you can run with `-d [QUESTION]` if you want a straight answer.

## Poisoning
Inside `malicious/`, insert the text files that you want to poison with (I've provided some samples)

### Testing Success Rate
Run `create_embeddings.py` to recreate embeddings with your new malicious text, then modify the top portions of `attack_evaluation.py` and then run it. It will create some statistics on poisoning success rate.
