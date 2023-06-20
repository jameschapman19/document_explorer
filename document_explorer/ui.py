import PySimpleGUI as sg


def create_ui():
    # Define the layout of the UI
    layout = [
        [sg.Text("Select a folder containing pdfs and/or word documents:")],  # A text label
        [
            sg.Input(key="-FOLDER-"),
            sg.FolderBrowse(),
        ],  # A text input box and a folder browse button
        [sg.Text("Enter a query:")],  # Another text label
        [sg.Input(key="-QUERY-")],  # A text input box for the query
        [sg.Button("Search"), sg.Button("Exit")],  # Two buttons
        [sg.Output(size=(80, 20))],  # An output area
    ]
    # Create the window object
    window = sg.Window("PDF Embedder", layout)
    # Return the window object
    return window
