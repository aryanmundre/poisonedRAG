import datasets
import os
import shutil
import math
import json
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

    documents = [{'text': doc, 'trust_score': 2} for doc in list(set(documents))]
    print(f"Loaded {len(documents)} docs from SQuAD...")

    # read malicious texts
    if os.path.exists('malicious.json'):
        with open('malicious.json', 'r') as f:
            data = json.loads(f.read())

            # change it to 'new' or 'old' based on which type of malicious texts you want
            for m in data['old']:
                documents.append({
                    'trust_score': 1,
                    'text': m,
                })

            print(f"Loaded {len(data['old'])} malicious text(s)")

    # Split all contexts into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=300,
    )
    split_texts = []
    for doc in documents:
        # Get the text and metadata
        text = doc["text"]
        metadata = doc["trust_score"]
        
        # Split the text into chunks
        chunks = text_splitter.split_text(text)
        
        # Attach the metadata to each chunk
        for chunk in chunks:
            split_texts.append({
                "text": chunk,
                "trust_score": metadata  # associate the same metadata with each chunk
            })

    # Initialize Chroma vector store (persistent storage)
    vector_store = Chroma(
        collection_name="my_collection",
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=embed_model,
    )
    # Add documents in batches to the vector store
    total_batches = math.ceil(len(split_texts) / 25)
    for batch in tqdm(batchify(split_texts, 25), total=total_batches, desc="Adding texts"):
        vector_store.add_texts(texts=[b['text'] for b in batch], metadatas=[{'trust_score': b['trust_score']} for b in batch])

    print(f"Embeddings created and stored in {PERSIST_DIRECTORY}.")        

if __name__ == '__main__':
    if os.path.exists(PERSIST_DIRECTORY):
        shutil.rmtree(PERSIST_DIRECTORY)
    create_embeddings()