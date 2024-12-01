# Instructions

### RAG Setup
Running `create_embeddings.py` will create the embeddings for the data in `knowledge_base/` and store them in `chroma_db/`. Run `create_embeddings.py --rerun` if you modify your knowledge base to regen the embeddings. Then you can run `main.py` to run an interactive prompt with ollama and you can run with `-d [QUESTION]` if you want a straight answer.

## Poisoning
Inside `knowledge_base/malicious`, insert the text files that you want to poison with (I've already provided a sample one messing with Lugia's Pokedex number and sample one with Squirtle's)

### Testing Success Rate
Run `create_embeddings.py --rerun` to recreate embeddings with your new malicious text, then modify the top portions of `attack_evaluation.py` and then run it. It will create some statistics on poisoning success rate.

### Some Sample Results
I poisoned Lugia differently than Squirtle (you can view the texts in the malicious knowledge base) and Lugia has a much much higher attack success rate (ASR) if you run the testing script setup for it. Take a look at the code to run it for yourself

### Note
You don't need to touch `format_knowledge_base.py`, I just included it cuz that's what I used to create files in `knowledge_base/`. Also don't need to touch `pokemon.csv` but that's again my original data document that you can look at if you're interested