# Import LangChain and other libraries
import langchain as lc
import os
import pandas as pd
from langchain.llms import HuggingFaceHub
from langchain.embeddings import HuggingFaceEmbeddings
import os
from getpass import getpass
HUGGINGFACEHUB_API_TOKEN = getpass()
os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGINGFACEHUB_API_TOKEN

# Initialize an HuggingFace LLM
repo_id = "google/flan-t5-xl" # See https://huggingface.co/models?pipeline_tag=text-generation&sort=downloads for some other options

llm = HuggingFaceHub(repo_id=repo_id, model_kwargs={"temperature":0, "max_length":64})
embedder = HuggingFaceEmbeddings()

# Define a function to load pdf documents from a folder
def load_pdfs(folder):
  # Create an empty list to store the documents
  docs = []
  # Loop through the files in the folder
  for file in os.listdir(folder):
    # Check if the file is a pdf
    if file.endswith(".pdf"):
      # Load the pdf using LangChain's document loader
      doc = lc.DocumentLoaders.PDF(os.path.join(folder, file))
      # Append the doc to the list
      docs.append(doc)
  # Return the list of documents
  return docs

# Define a function to generate embeddings for the documents using OpenAI's language model
def embed_docs(docs):
  # Create an empty list to store the embeddings
  embeddings = []
  # Loop through the documents
  for doc in docs:
    # Generate an embedding using OpenAI's Embeddings API
    embedding = lc.Embeddings.OpenAI(doc.text)
    # Append the embedding to the list
    embeddings.append(embedding)
  # Return the list of embeddings
  return embeddings


# Define a function to find the nearest document to a query in embedding space
def find_nearest_doc(query, docs, embeddings):
  # Generate an embedding for the query using OpenAI's Embeddings API
  query_embedding = embedder(query)
  # Compute the cosine similarity between the query embedding and each document embedding
  similarities = [lc.Utils.cosine_similarity(query_embedding, doc_embedding) for doc_embedding in embeddings]
  # Find the index of the document with the highest similarity
  index = similarities.index(max(similarities))
  # Return the document at that index
  return docs[index]

# Define a function to answer a query using the nearest document
def answer_query(query, docs, embeddings):
  # Find the nearest document to the query
  nearest_doc = find_nearest_doc(query, docs, embeddings)
  # Construct a prompt for OpenAI's language model to answer the query using the document text
  prompt = f"Q: {query}\nA: {nearest_doc.text}\nAnswer:"
  # Generate a response using OpenAI's language model with a stop sequence of "\n"
  response = llm.generate(prompt, stop="\n")
  # Return the response
  return response
