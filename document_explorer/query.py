# Import LangChain and other libraries
import os
import re
from getpass import getpass
from langchain import PromptTemplate
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFaceHub
import torch
# from langchain.llms import HuggingFaceHub, OpenAI
from langchain.vectorstores import FAISS, ElasticVectorSearch

# load environment variables
load_dotenv()

class QueryEngine:
    def __init__(
        self,
        llm_type="HuggingFaceHub",
        llm_repo_id="google/flan-t5-xxl",
        embedding_type="HuggingFaceEmbeddings",
        embedding_model_name="sentence-transformers/all-mpnet-base-v2",
    ):
        # Initialize the LLM and the embedding generator using the factory functions
        self.llm = create_llm(llm_type, llm_repo_id)
        self.embeddings = create_embedding_generator(
            embedding_type, embedding_model_name
        )
        # Initialize an empty vector store
        self.vector_store = None

    def embed_docs(self, documents, vector_store_type="FAISS"):
        # Generate embeddings for each document using the factory function
        self.vector_store = create_vector_store(
            vector_store_type, documents, self.embeddings
        )
        self.vector_store.save_local("vector_store")
        self.vector_store = FAISS.load_local("vector_store", self.embeddings)


    def ask_question(self, query):
        # Check if the vector store is not empty
        if self.vector_store:
            # Create a retriever from the vector store
            retriever = self.vector_store.as_retriever(
                search_type="similarity", search_kwargs={"k": 1}
            )
            # Create a question answering chain from the LLM and the retriever
            qa = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=retriever,
                return_source_documents=True,
            )

            # Ask the question using the chain
            result = qa({"query": query})
            # Return the result
            return result
        else:
            # Raise an exception if the vector store is empty
            raise ValueError("No documents have been embedded")


def create_vector_store(vector_store_type, documents, embedding_generator):
    # Create a dictionary of vector store types and classes
    vector_stores = {
        "FAISS": FAISS,
        "ElasticVectorSearch": ElasticVectorSearch,
        # Add more as needed
    }
    # Check if the vector store type is supported
    if vector_store_type in vector_stores:
        # Return an instance of the corresponding vector store class
        return vector_stores[vector_store_type].from_documents(
            documents, embedding_generator
        )
    else:
        # Raise an exception if the vector store type is not supported
        raise ValueError(f"Unsupported vector store type: {vector_store_type}")


def create_llm(llm_type, llm_repo_id):
    # Create a dictionary of llm types and classes
    llms = {
        "HuggingFaceHub": HuggingFaceHub,
        # Add more as needed
    }
    # Check if the llm type is supported
    if llm_type in llms:
        # Return an instance of the corresponding llm class
        return llms[llm_type](repo_id=llm_repo_id,
                              model_kwargs={"temperature":0.1,  "max_length":64})
    else:
        # Raise an exception if the llm type is not supported
        raise ValueError(f"Unsupported llm type: {llm_type}")


def create_embedding_generator(embedding_type, embedding_model_name):
    # Create a dictionary of embedding types and classes
    embeddings = {
        "HuggingFaceEmbeddings": HuggingFaceEmbeddings,
        # Add more as needed
    }
    # Check if the embedding type is supported
    if embedding_type in embeddings:
        # Return an instance of the corresponding embedding class
        return embeddings[embedding_type](
            model_name=embedding_model_name,
             model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
            encode_kwargs={"normalize_embeddings": False},
        )
    else:
        # Raise an exception if the embedding type is not supported
        raise ValueError(f"Unsupported embedding type: {embedding_type}")
