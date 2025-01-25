from utils.constants import vector_search_prompt
import os
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from datetime import date

from dotenv import load_dotenv
load_dotenv()


class VectorSearch:
    def __init__(self):
        self.embeddings = NVIDIAEmbeddings(
            model="nvidia/llama-3.2-nv-embedqa-1b-v2",
            api_key=os.getenv("NVIDIA_API_KEY"),
            truncate="NONE",
        )
        self.vector_db = Chroma(
            embedding_function=self.embeddings, persist_directory="./vector_db")
        self.llm = ChatGroq(model='llama-3.3-70b-versatile')

    def search(self, input):
        # Create a retriever from the vectorstore
        retriever = self.vector_db.as_retriever(search_kwargs={"k": 1})

        # Create the RetrievalQA chain
        combine_docs_chain = create_stuff_documents_chain(
            llm=self.llm, prompt=vector_search_prompt
        )
        qa_chain = create_retrieval_chain(
            retriever=retriever,
            combine_docs_chain=combine_docs_chain
        )

        # Perform semantic search using the refined query
        results = qa_chain.invoke({
            "input": input,
            "current_date": date.today().strftime("%d-%m-%y"),
        })

        return results['answer']
