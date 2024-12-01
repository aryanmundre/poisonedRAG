import datasets
import os
import shutil
import math
from tqdm import tqdm
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load the embeddings model
embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Define folder path containing Chroma's persistent storage path
PERSIST_DIRECTORY = "chroma_db"  # Directory to store/load embeddings

def batchify(data, batch_size):
    """Yield successive batches from data."""
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size]

def create_embeddings():
    # load SQUAD dataset...first 200 of them
    dataset = datasets.load_dataset('squad')['train'][:500]

    # get document list based on training set's contexts
    documents = []
    for context in dataset['context']:
        documents.append(context)
    documents = list(set(documents))
    print(f"Loading {len(documents)} docs from SQuAD...")

    # read malicious texts
    for filename in os.listdir(f'malicious'):
        if filename.endswith(".txt"):
            file_path = os.path.join(f'malicious', filename)
            print(f'Found malicious text {file_path}')
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
                documents.append(text)

    # Split all contexts into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=300,
    )
    split_texts = []
    for context in documents:
        split_texts.extend(text_splitter.split_text(context))

    # Initialize Chroma vector store (persistent storage)
    vector_store = Chroma(
        collection_name="my_collection",
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=embed_model,
    )
    # Add documents in batches to the vector store
    total_batches = math.ceil(len(split_texts) / 25)
    for batch in tqdm(batchify(split_texts, 25), total=total_batches, desc="Adding texts"):
        vector_store.add_texts(texts=batch)

    print(f"Embeddings created and stored in {PERSIST_DIRECTORY}.")
    

if __name__ == '__main__':
    if os.path.exists(PERSIST_DIRECTORY):
        shutil.rmtree(PERSIST_DIRECTORY)
    create_embeddings()