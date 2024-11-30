import os
import sys
from langchain_chroma import Chroma
import shutil
from langchain_huggingface import HuggingFaceEmbeddings

# Load the embeddings model
embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Define folder path containing the documents and Chroma's persistent storage path
KNOWLEDGE_BASE = "knowledge_base"  # Change this to your actual folder path
PERSIST_DIRECTORY = "chroma_db"  # Directory to store/load embeddings

def batchify(data, batch_size):
    """Yield successive batches from data."""
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size]

def create_embeddings():
    # Check if the Chroma database already exists
    if os.path.exists(PERSIST_DIRECTORY):
        print("Embeddings already exist. Skipping embedding creation.")
        return

    # read a data set
    num_processed = 0
    all_texts = []
    for filename in os.listdir(f'{KNOWLEDGE_BASE}/data'):
        if filename.endswith(".txt"):
            file_path = os.path.join(f'{KNOWLEDGE_BASE}/data', filename)

            num_processed += 1

            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
                all_texts.append(text)
    print(f'Processed {num_processed} texts')

    # read malicious texts
    for filename in os.listdir(f'{KNOWLEDGE_BASE}/malicious'):
        if filename.endswith(".txt"):
            file_path = os.path.join(f'{KNOWLEDGE_BASE}/malicious', filename)
            print(f'Found malicious text {file_path}')
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
                all_texts.append(text)

    # Create vector database from the chunks using embeddings and store it persistently
    # Initialize the vector store (e.g., Chroma) and split chunks into smaller batches
    for batch in batchify(all_texts, 100):
        vector_store = Chroma.from_texts(
            texts=batch,  # Pass a batch of text
            embedding=embed_model,
            persist_directory="chroma_db"
        )

    print(f"Embeddings created and stored in {PERSIST_DIRECTORY}.")

if __name__ == "__main__":
    if os.path.exists(PERSIST_DIRECTORY) and len(sys.argv) > 1 and sys.argv[1] == '--rerun':
        shutil.rmtree(PERSIST_DIRECTORY)
    create_embeddings()