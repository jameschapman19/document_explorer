# Import PySimpleGUI
import PySimpleGUI as sg

from querya.pdf_embedder import answer_query, load_pdfs, embed_docs


# Define the layout of the UI
layout = [
    [sg.Text("Select a folder containing pdfs:")], # A text label
    [sg.Input(key="-FOLDER-"), sg.FolderBrowse()], # A text input box and a folder browse button
    [sg.Text("Enter a query:")], # Another text label
    [sg.Input(key="-QUERY-")], # A text input box for the query
    [sg.Button("Search"), sg.Button("Exit")], # Two buttons
    [sg.Output(size=(80, 20))] # An output area
]

# Create the window object
window = sg.Window("PDF Embedder", layout)

# Create an event loop
while True:
    event, values = window.read() # Read user input
    if event == "Exit" or event == sg.WIN_CLOSED: # If user closes window or clicks exit
        break # Exit the loop
    elif event == "Search": # If user clicks search
        folder = values["-FOLDER-"] # Get the folder path from the input box
        query = values["-QUERY-"] # Get the query from the input box
        if folder and query: # If both are not empty
            docs = load_pdfs(folder) # Load the pdf documents from the folder
            embeddings = embed_docs(docs) # Generate embeddings for the documents 
            answer = answer_query(query, docs, embeddings) # Call your function to answer the query with the docs and embeddings arguments
            print(answer) # Print the answer to the output area

# Close the window
window.close()
