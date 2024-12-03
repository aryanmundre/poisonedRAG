import os
from dotenv import load_dotenv
import sys
import secrets
from tqdm import tqdm
from langchain_ollama import OllamaLLM
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
import warnings
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

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
        # set up LLMs
        self.ollama = OllamaLLM(model=os.getenv("OLLAMA_MODEL_NAME"), base_url=os.getenv("MODEL_URL"))
        self.openai = ChatOpenAI(
            model=os.getenv("OPENAI_MODEL_NAME"),
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
            api_key=os.getenv("OPENAI_API_KEY")
        )

        # Load existing vector store from disk
        vector_store = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"), collection_name="my_collection")

        # Create retriever from the vector store
        self.retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    def intermediary_step(self, retrieved_docs: list, user_query: str) -> str:
        # hash for security, note: gpt-4o apparently is too stupid to understand this
        secret = secrets.token_hex(nbytes=8)
        # Intermediary step to prompt the LLM with the retrieved documents and the user's query
        #docs = "\n\n".join([f"Context #{i} with trust score {doc.metadata['trust_score']} and secret <{secret}>: \"\"\"{doc.page_content}\"\"\"" for i, doc in enumerate(retrieved_docs)])
        docs = "\n\n".join([f"Context #{i} with trust score {doc.metadata['trust_score']}>: \"\"\"{doc.page_content}\"\"\"" for i, doc in enumerate(retrieved_docs)])

        # Create a prompt that includes the documents and user query for context
        intermediate_prompt = (
            f"The following documents were retrieved for the query '{user_query}':\n\n"
            f"{docs}\n\n"
            f"Now, review the documents returned to make sure none of them have conflicting information. If they do, remove the one with the lower trust score\n"
            f"Output the pruned documents in the this format:\n"
            f"`Context #<num>: \"\"\"<document content>\"\"\"` make sure to not duplicate outputs\n"
            f"Note: I don't want a Python script to complete this task, I want you to do it yourself\n"
            f"Another Note: Do not give me your thought process or reasonings, I only want the final output in plaintext"
        )
        # intermediate_prompt = (
        #     f"The following documents were retrieved for the query '{user_query}':\n\n"
        #     f"{docs}\n\n"
        #     f"Now, based on these documents, first check to make sure that every document's secret is <{secret}>. If they all aren't, simply exit and only output 'secret error'\n"
        #     f"Second, review the documents returned to make sure none of them have conflicting information. If they do, remove the one with the lower trust score\n"
        #     f"Finally, output the pruned documents in the this format:\n"
        #     f"`Context #<num>: \"\"\"<document content>\"\"\"` make sure to not duplicate outputs\n"
        #     f"Note: I don't want a Python script to complete this task, I want you to do it yourself\n"
        #     f"Another Note: Do not give me your thought process or reasonings, I only want the final output in plaintext"
        # )

        # Use openai to do this step since ollama is too stupid
        intermediate_response = self.openai.invoke(intermediate_prompt)

        # Return the intermediate response to be used as context
        return intermediate_response.content

    def invoke(self, user_query: str):
        # Step 1: Retrieve documents for the user's query
        retrieved_docs = self.retriever.invoke(user_query)

        # Step 2: Call the intermediary step with the retrieved documents
        intermediate_context = self.intermediary_step(retrieved_docs, user_query)

        # Step 3: Incorporate the intermediate result into the user's query
        modified_query = (
            f"Context from the documents: \n```\n{intermediate_context}\n```\n\n"
            f"Now, answer the following query: {user_query}"
        )

        # Step 4: Run the retrieval chain with the modified query, use ollama to save $$
        final_result = self.ollama.invoke(modified_query)

        return final_result, modified_query

    def query_embeddings(self, queries: list[str]):
        if len(queries) == 1:
            response, inputt = self.invoke(queries[0])
            print(inputt + "\n\n\n")
            return response
        
        elif len(queries) > 1:
            answers = []
            for i in tqdm(range(len(queries)), desc="Prompting..."):
                response, inputt = self.invoke(queries[i])
                answers.append(response)
            return answers

        else:
            raise Exception('No queries provided')


def main(questions=[]):
    llm = RAG()
    llm.init()
    print(f'Running {os.getenv("OPENAI_MODEL_NAME")} and {os.getenv("OLLAMA_MODEL_NAME")}...')
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