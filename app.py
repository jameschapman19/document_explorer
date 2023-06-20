import PySimpleGUI as sg

# Import the create_ui function
from document_explorer.document_loading import load_docs

# Import the QueryEngine class
from document_explorer.query import QueryEngine
from document_explorer.ui import create_ui

# Create an instance of the QueryEngine class
query_engine = QueryEngine(
    llm_type="HuggingFaceHub",
    llm_repo_id="google/flan-t5-xxl",
    embedding_type="HuggingFaceEmbeddings",
    embedding_model_name="sentence-transformers/all-mpnet-base-v2",
)


# Create the UI window
window = create_ui()

# Create an event loop
while True:
    event, values = window.read()  # Read user input
    if (
        event == "Exit" or event == sg.WIN_CLOSED
    ):  # If user closes window or clicks exit
        break  # Exit the loop
    elif event == "Search":  # If user clicks search
        folder = values["-FOLDER-"]  # Get the folder path from the input box
        query = values["-QUERY-"]  # Get the query from the input box
        if folder and query:  # If both are not empty
            documents = load_docs(folder)  # Load the documents from the folder
            query_engine.embed_docs(
                documents
            )  # Embed the documents using the query engine
            answer = query_engine.ask_question(
                query
            )  # Ask the question using the query engine
            print("answer",answer['result'])  # Print the answer to the output area
            print("source",answer['source_documents'])  # Print the source to the output area

# Close the window
window.close()
