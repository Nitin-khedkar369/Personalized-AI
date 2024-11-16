# Import the dependent libraries
import chromadb
import ollama
from transformers import LlamaTokenizer
# Importing libraries to PDF and Doc
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import UnstructuredWordDocumentLoader
# Importing libraries to split
from langchain.text_splitter import RecursiveCharacterTextSplitter
# Importing libraries to multi-threading
from concurrent.futures import ThreadPoolExecutor
import time
# Importing library to check the cpu in your system
import os

# Document path where the file exists
# Mine is in data folder hence data/file name
DATA_PATH = "data/harry_potter_and_the_deathly_hallows.pdf"


# Load the required documents
# To Load PDF
document_loader = PyPDFLoader(DATA_PATH)
documents = document_loader.load()

# To Load Word docs
# document_loader = UnstructuredWordDocumentLoader(DATA_PATH)
# documents = document_loader.load()


# Defining the Split i.e how to split,chunk size etc
text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
    chunk_overlap=100,
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



# Split the document
splits = text_splitter.split_documents(documents)


# Initialize ChromaDB collection
client = chromadb.Client()
collection = client.create_collection(name='docs')


max_iterations = 5

# Function to process a single document chunk
# Also don't forget to put your llm model downloaded locally for example I have used llama3.1:8b
def process_document_chunk(i, chunk):
    try:
        response = ollama.embeddings(model="llama3.1:8b", prompt=chunk.page_content)
        embedding = response["embedding"]
        collection.add(
            ids=[str(i)],
            embeddings=[embedding],
            documents=[chunk.page_content]
        )
        print(f"Processed chunk {i}")
    except Exception as e:
        print(f"Error processing chunk {i}: {e}")

# Run the below code to check the max CPU core we can use for parallel processing
# To check Max worker in your CPU
# max_workers = os.cpu_count()
# print(f"Using {max_workers} workers.")

# Ideal number of CPU to balance between system load and thread utility.
max_workers = min(32, (os.cpu_count() or 1) + 4)
# print(f"Using Ideal number of {max_workers} workers.")

# Using ThreadPoolExecutor for parallel processing
start_time = time.time()
with ThreadPoolExecutor(max_workers=max_workers) as executor:
    for i, chunk in enumerate(splits):
        if i >= max_iterations:
            break  # Stop after max_iterations
        executor.submit(process_document_chunk, i, chunk)

print(f"Processing completed in {time.time() - start_time} seconds.")



# Example prompt
prompt = ""
while prompt.lower() not in ["bye", "b", "exit"]:

    prompt = input("Prompt: ")
    if prompt.lower() in ["bye", "b", "exit"]:
        print("ADIOS Amigos thanks for interacting with me see you later!")
        break


# Generate an embedding for the prompt and retrieve the most relevant doc
    response = ollama.embeddings(
        prompt=prompt,
        model="llama3.1:8b"
    )

    results = collection.query(
        query_embeddings=[response["embedding"]],
        n_results=1
    )

    data = results["documents"][0][0]

# Generate a response combining the prompt and data we retrieved in step 2
    output = ollama.generate(
# Make sure the model name is correctly specified
        model="llama3.1:8b",
        prompt=f"Using this data: {data}. Response to this prompt: {prompt}"
    )

    print(output['response'])
