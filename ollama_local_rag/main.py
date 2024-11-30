import os
import sys
from langchain_ollama import OllamaLLM
from langchain_chroma import Chroma
from langchain.chains import create_retrieval_chain
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
import warnings
from sentence_transformers import SentenceTransformer
from langchain_huggingface import HuggingFaceEmbeddings

# Suppress specific LangSmithMissingAPIKeyWarning
warnings.filterwarnings("ignore", category=UserWarning, module="langsmith")

# Initialize models
llm = OllamaLLM(model="llama3.2", base_url="http://127.0.0.1:11434")

# init embed model
# Load the SentenceTransformer model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Define the Chroma persistent storage path
PERSIST_DIRECTORY = "chroma_db"  # Directory to store/load embeddings

def query_embeddings(query):
    # Check if the embeddings have been created and stored
    if not os.path.exists(PERSIST_DIRECTORY):
        raise Exception("Embeddings not found. Please run create_embeddings.py first.")

    # Load existing vector store from disk
    vector_store = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"))

    # Create retriever from the vector store
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    # Initialize the retrieval-qa-chat prompt
    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")

    # Combine documents using a Stuff Documents chain
    combine_docs_chain = create_stuff_documents_chain(
        llm, retrieval_qa_chat_prompt
    )

    # Create the retrieval chain
    retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)

    # Query the retrieval chain to find the top 5 documents returned
    docs = retriever.invoke(query)
    for doc in docs:
        print(doc)
    response = retrieval_chain.invoke({"input": query})
    return response['answer']

def main(questions=[]):
    # this if statement is used when this is called as a module rather than ran directly
    answers = []
    if len(questions) > 0:
        for question in questions:
            answer.append(query_embeddings(question))
        return answers
            

    if len(sys.argv) > 1 and sys.argv[1] == '-d':
        answer = query_embeddings(' '.join(sys.argv[2:]))
        print("Answer:", answer)
    else:
        while True:
            query = input("> ")
            if query == "/bye":
                exit()
            
            answer = query_embeddings(query)
            print("Answer:", answer)

if __name__ == "__main__":
    main()
        