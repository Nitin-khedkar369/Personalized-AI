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
import streamlit as st

# Path to the folder containing files
DATA_PATH = "data/"

# Initialize ChromaDB collection
client = chromadb.Client()

collection_name = 'docs'
try:
    # Check if the collection exists, then retrieve or create
    if collection_name in [col.name for col in client.list_collections()]:
        collection = client.get_collection(name=collection_name)
        print(f"Retrieved existing collection: {collection_name}")
    else:
        collection = client.create_collection(name=collection_name)
        print(f"Created new collection: {collection_name}")
except Exception as e:
    print(f"Error initializing collection: {e}")

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
        return None
    return files


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
        separators=["\n\n", "\n", " ", ".", ",", "\u200b", "\uff0c", "\u3001", "\uff0e", "\u3002", ""]
    )
    splits = text_splitter.split_documents(documents)
    return splits, chunk_size


# Function to process a single document chunk
def process_document_chunk(i, chunk):
    try:
        response = ollama.embeddings(model="nomic-embed-text", prompt=chunk.page_content)
        embedding = response["embedding"]
        collection.add(
            ids=[str(i)],
            embeddings=[embedding],
            documents=[chunk.page_content]
        )
    except Exception as e:
        print(f"Error processing chunk {i}: {e}")


# Streamlit Frontend for the app
def main():
    st.title("AI Document Assistant")

    # Let the user upload a file
    uploaded_file = st.file_uploader("Upload a document", type=["pdf", "docx", "doc"])

    if uploaded_file is not None:
        st.write(f"Processing file: {uploaded_file.name}")

        # Save uploaded file to disk for processing
        file_path = os.path.join(DATA_PATH, uploaded_file.name)

        # Load and split the selected document
        splits, chunk_size = load_and_split_document(file_path)

        # Set the max iterations based on chunk size
        max_iterations = 5 if chunk_size < 1200 else 9

        # Use ThreadPoolExecutor for parallel processing
        max_workers = min(32, (os.cpu_count() or 1) + 4)
        start_time = time.time()
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for i, chunk in enumerate(splits):
                if i > max_iterations:
                    break  # Stop after max_iterations
                executor.submit(process_document_chunk, i, chunk)
        st.success(f"Processing completed in {time.time() - start_time:.2f} seconds.")

    # Interactive Q&A
    st.header("Ask Questions")
    prompt = st.chat_input("Enter your query:")

    if prompt:
        # Generate an embedding for the prompt
        response = ollama.embeddings(prompt=prompt, model="nomic-embed-text")

        # Retrieve the most relevant document
        results = collection.query(query_embeddings=[response["embedding"]], n_results=1)

        if results["documents"]:
            data = results["documents"][0][0]
            output = ollama.generate(
                model="llama3.2:latest",
                prompt=f"Using this data: {data}. Respond to this prompt: {prompt}"
            )
            message = st.chat_message("assistant")
            message.write(output['response'])
        else:
            st.warning("No relevant data found to answer your query.")


if __name__ == "__main__":
    main()
