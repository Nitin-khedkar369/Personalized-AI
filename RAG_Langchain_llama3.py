# Import the dependent libraries
import chromadb
import ollama
from langchain_community.document_loaders import (
    PyPDFLoader,
    UnstructuredWordDocumentLoader,
    DirectoryLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from concurrent.futures import ThreadPoolExecutor
import os
import time
import keyboard

# Path to the folder containing files
DATA_PATH = "data/"

# Initialize ChromaDB collection
client = chromadb.Client()
collection = client.create_collection(name='docs')


def determine_chunk_parameters(documents, max_chunks=1000):
    """
    Automatically determine chunk size and overlap based on the document properties.

    Args:
        documents (list): A list of document objects with content.
        max_chunks (int): Maximum number of chunks desired.

    Returns:
        tuple: Optimal chunk size and chunk overlap.
    """
    total_content = "".join([doc.page_content for doc in documents])
    total_length = len(total_content)

    # Automatically calculate chunk size to fit within max_chunks
    optimal_chunk_size = max(1000, min(total_length // max_chunks, 3000))

    # Set chunk overlap as 10% of the chunk size, ensuring minimum overlap
    optimal_chunk_overlap = max(100, optimal_chunk_size // 10)

    return optimal_chunk_size, optimal_chunk_overlap


# Function to list supported files in the directory
def list_files(folder_path):
    files = [f for f in os.listdir(folder_path) if f.endswith((".pdf", ".docx", ".doc"))]
    if not files:
        print("No supported files found in the directory.")
        return None
    print("Available files:")
    for idx, file in enumerate(files, 1):
        print(f"{idx}. {file}")
    return files


# Function to select a file
def select_file(files):
    while True:
        try:
            choice = int(input("Select a file by entering its number: "))
            if 1 <= choice <= len(files):
                return files[choice - 1]
            else:
                print("Invalid choice. Please try again.")
        except ValueError:
            print("Please enter a valid number.")


# Function to load and split a selected document
def load_and_split_document(file_path):
    extension = os.path.splitext(file_path)[1].lower()
    if extension == ".pdf":
        loader = PyPDFLoader(file_path)
    elif extension in [".docx", ".doc"]:
        loader = UnstructuredWordDocumentLoader(file_path)
    else:
        raise ValueError(f"Unsupported file format: {extension}")

    documents = loader.load()

    # Determine optimal chunk size and overlap
    chunk_size, chunk_overlap = determine_chunk_parameters(documents)

    # Define the text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=[
            "\n\n",
            "\n",
            " ",
            ".",
            ",",
            "\u200b",  # Zero-width space
            "\uff0c",  # Full width comma
            "\u3001",  # Ideographic comma
            "\uff0e",  # Full width full stop
            "\u3002",  # Ideographic full stop
            "",
        ]
    )
    print(f"Chunk Size: {chunk_size}, Chunk Overlap: {chunk_overlap}")
    splits = text_splitter.split_documents(documents)
    return splits, chunk_size


# Function to process a single document chunk
def process_document_chunk(i, chunk):
    try:
        response = ollama.embeddings(model="llama3.2:latest", prompt=chunk.page_content)
        embedding = response["embedding"]
        collection.add(
            ids=[str(i)],
            embeddings=[embedding],
            documents=[chunk.page_content]
        )
        print(f"Processed chunk {i}")
    except Exception as e:
        print(f"Error processing chunk {i}: {e}")


# Main logic
def main():
    # List available files
    files = list_files(DATA_PATH)
    if not files:
        return

    # Let the user select a file
    selected_file = select_file(files)
    file_path = os.path.join(DATA_PATH, selected_file)
    print(f"Selected file: {selected_file}")

    # Load and split the selected document
    splits, chunk_size = load_and_split_document(file_path)

    if chunk_size < 1200:
        max_iterations = 5
    else:
        max_iterations = 9

    # Use ThreadPoolExecutor for parallel processing
    max_workers = min(32, (os.cpu_count() or 1) + 4)
    start_time = time.time()
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for i, chunk in enumerate(splits):
            if i > max_iterations:
                break # Stop after max_iterations
            executor.submit(process_document_chunk, i, chunk)
    print(f"Processing completed in {time.time() - start_time} seconds.")

    # Interactive Q&A loop
    prompt = ""
    print("To exit type 'bye' or 'exit' in the prompt.")

    while True:
        prompt = input("Prompt: ")

        if prompt.lower() in ["bye", "exit"]:
            print("ADIOS Amigos! Thanks for interacting with me. See you later!")
            break

        # Generate an embedding for the prompt
        response = ollama.embeddings(prompt=prompt, model="llama3.2:latest")

        # Retrieve the most relevant document
        results = collection.query(query_embeddings=[response["embedding"]], n_results=1)
        data = results["documents"][0][0]

        # Generate a response combining the prompt and the retrieved document
        output = ollama.generate(
            model="llama3.2:latest",
            prompt=f"Using this data: {data}. Respond to this prompt: {prompt}"
        )

        print(output['response'])


if __name__ == "__main__":
    main()


