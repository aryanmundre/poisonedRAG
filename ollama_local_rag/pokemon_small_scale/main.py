import os
from dotenv import load_dotenv
import sys
from tqdm import tqdm
from langchain_ollama import OllamaLLM
from langchain_chroma import Chroma
from langchain.chains import create_retrieval_chain
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
import warnings
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()
MODEL_NAME = os.getenv('MODEL_NAME')
MODEL_URL = os.getenv('MODEL_URL')

# Suppress specific LangSmithMissingAPIKeyWarning
warnings.filterwarnings("ignore", category=UserWarning, module="langsmith")

# Define the Chroma persistent storage path
PERSIST_DIRECTORY = "chroma_db"  # Directory to store/load embeddings

class RAG():
    def __init__(self) -> None:
        # Check if the embeddings have been created and stored
        if not os.path.exists(PERSIST_DIRECTORY):
            raise Exception("Embeddings not found. Please run create_embeddings.py first.")

    def init(self) -> None:
        # set up Ollama
        llm = OllamaLLM(model=MODEL_NAME, base_url=MODEL_URL)

        # Load existing vector store from disk
        vector_store = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"), collection_name="my_collection")

        # Create retriever from the vector store
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})

        # Initialize the retrieval-qa-chat prompt
        retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")

        # Combine documents using a Stuff Documents chain
        combine_docs_chain = create_stuff_documents_chain(
            llm, retrieval_qa_chat_prompt
        )

        # Create the retrieval chain
        self.retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)

    def query_embeddings(self, queries: list[str]):
        if len(queries) == 1:
            response = self.retrieval_chain.invoke({"input": queries[0]})
            for doc in response['context']:
                print(doc)
            print()
            return response['answer']
        
        elif len(queries) > 1:
            answers = []
            for i in tqdm(range(len(queries)), desc="Prompting..."):
                response = self.retrieval_chain.invoke({"input": queries[i]})
                answers.append(response['answer'])
            return answers

        else:
            raise Exception('No queries provided')


def main(questions=[]):
    llm = RAG()
    llm.init()
    # this if statement is used when this script is called as a module rather than ran directly
    if len(questions) > 0:
        return llm.query_embeddings(questions)
        
    # "interactive" modes
    if len(sys.argv) > 1 and sys.argv[1] == '-d':
        answer = llm.query_embeddings([' '.join(sys.argv[2:])])
        print("Answer:", answer)
    else:
        while True:
            query = input("> ")
            if query == "/bye":
                exit()
            
            answer = llm.query_embeddings([query])
            print("Answer:", answer)

if __name__ == "__main__":
    main()