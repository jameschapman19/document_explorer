# Document Explorer

This app allows you to ask natural language questions about a folder of documents and get answers from the most relevant document. It uses a language model and an embedding generator to perform retrieval-based question answering.

## Features

- Document Explorer allows you to query PDF documents using natural language, without having to open or read them manually.
- Document Explorer uses LangChain, a framework for developing applications powered by language models, to generate embeddings and answers for the documents and queries.
- Document Explorer supports various language models from HuggingFaceHub, such as Google's FLAN-T5-XL, which can handle multiple languages and domains.
- Document Explorer has a simple and intuitive graphical user interface, built with PySimpleGUI, that lets you interact with your documents easily.

## Requirements

- Python 3.8 or higher
- PySimpleGUI
- langchain
- pandas
- getpass
- unstructured
- huggingface_hub
- sentence_transformers

## Installation

To install the app, clone this repository and run the following command in the terminal:

```
pip install -r requirements.txt
```

You will also need to create a .env file in the root directory of the app and add your HuggingFaceHub API token as follows:

HUGGINGFACEHUB_API_TOKEN=your_token_here

## Usage

To run the app, run the following command in the terminal:

```
python app.py
```

This will open a graphical user interface where you can select a folder of documents and enter a query. The app supports documents in .doc, .docx, .pdf, and .html formats.

The app will load and split the documents into chunks, embed them using a sentence transformer model, and store them in a FAISS vector store. Then, it will use a T5 model from HuggingFaceHub to generate an answer from the most similar document chunk.

The app will display the answer and the source document in the output area.
