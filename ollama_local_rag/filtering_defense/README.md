# Instructions

### RAG Setup
Running `create_embeddings.py` will create the embeddings for the data pulled from SQuAD dataset and store them in `chroma_db/`. Then you can run `main.py` to run an interactive prompt with ollama and you can run with `-d [QUESTION]` if you want a straight answer. You will also need an openai api key in `.env`

## Poisoning
Use the `malicious.json` file to insert poisoned docs. The 'new' key denotes poisoned docs with my scheme, while the 'old' key denotes poisoned docs with the original paper's black box scheme.

### Testing Success Rate
Run `attack_evaluation.py`. The `.env` file has parameters for the attack script. It will create some statistics on poisoning success rate.
