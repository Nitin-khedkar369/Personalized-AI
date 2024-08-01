# Import the dependent libraries
import chromadb
import ollama
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from concurrent.futures import ThreadPoolExecutor
import time

# Document path where the file exists
DATA_PATH = "data/Fund Management Policy Guidelines and Procedure.pdf"
# DATA_PATH = "data/monopoly.pdf"
# DATA_PATH = "data/SIC - 01 - 03 - 01 - Manage Inadequate Seed Money .pdf"


# Load the required documents
document_loader = PyPDFLoader(DATA_PATH)
documents = document_loader.load()


# Defining the Split i.e how to split,chunk size etc
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=90,
    separators=[
            "\n\n",
            "\n",
            " ",
            ".",
            ",",
            "\u200b",  # Zero-width space
            "\uff0c",  # Fullwidth comma
            "\u3001",  # Ideographic comma
            "\uff0e",  # Fullwidth full stop
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
# Also don't forget to put your llm model downloaded locally
def process_document_chunk(i, chunk):
    try:
        response = ollama.embeddings(model="llama3", prompt=chunk.page_content)
        embedding = response["embedding"]
        collection.add(
            ids=[str(i)],
            embeddings=[embedding],
            documents=[chunk.page_content]
        )
        print(f"Processed chunk {i}")
    except Exception as e:
        print(f"Error processing chunk {i}: {e}")

# Using ThreadPoolExecutor for parallel processing
start_time = time.time()
with ThreadPoolExecutor(max_workers=5) as executor:
    for i, chunk in enumerate(splits):
        if i >= max_iterations:
            break  # Stop after max_iterations
        executor.submit(process_document_chunk, i, chunk)

print(f"Processing completed in {time.time() - start_time} seconds.")


# Example prompt
# prompt = "What animals are llamas related to?"
# prompt = "What is ADSE?"
prompt = "summaries the entire document for me?"

# Generate an embedding for the prompt and retrieve the most relevant doc
response = ollama.embeddings(
    prompt=prompt,
    model="llama3"
)

results = collection.query(
    query_embeddings=[response["embedding"]],
    n_results=1
)

data = results["documents"][0][0]

# Generate a response combining the prompt and data we retrieved in step 2
output = ollama.generate(
    model="llama3"

,  # Make sure the model name is correctly specified
    prompt=f"Using this data: {data}. Response to this prompt: {prompt}"
)

print(output['response'])































# import chromadb
# import ollama
# import json
# from langchain_community.document_loaders import PyPDFLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from concurrent.futures import ThreadPoolExecutor
# import time
#
# DATA_PATH = "data/SIC - 01 - 03 - 01 - Manage Inadequate Seed Money .pdf"
#
# # Load the document
# document_loader = PyPDFLoader(DATA_PATH)
# documents = document_loader.load()
#
# # Initialize ChromaDB client and create a collection
# client = chromadb.Client()
# collection = client.create_collection(name='docs')
#
# # Initialize the text splitter
# text_splitter = RecursiveCharacterTextSplitter(
#     chunk_size=100,
#     chunk_overlap=10,
#     length_function=len,
#     is_separator_regex=False,
#     separators=[
#         "\n\n",
#         "\n",
#         " ",
#         ".",
#         ",",
#         "\u200b",  # Zero-width space
#         "\uff0c",  # Fullwidth comma
#         "\u3001",  # Ideographic comma
#         "\uff0e",  # Fullwidth full stop
#         "\u3002",  # Ideographic full stop
#         "",
#     ]
# )
#
# # Split the documents into chunks
# split_documents = []
# for doc in documents:
#     splits = text_splitter.split_documents([doc])
#     split_documents.extend(splits)
#
# max_iterations = 5
#
# # Function to process a single document chunk
# def process_document_chunk(i, chunk):
#     try:
#         response = ollama.embeddings(model="llama3", prompt=chunk.page_content)
#         embedding = response["embedding"]
#         collection.add(
#             ids=[str(i)],
#             embeddings=[embedding],
#             documents=[chunk.page_content]
#         )
#         print(f"Processed chunk {i}")
#     except Exception as e:
#         print(f"Error processing chunk {i}: {e}")
#
# # Using ThreadPoolExecutor for parallel processing
# start_time = time.time()
# with ThreadPoolExecutor(max_workers=5) as executor:
#     for i, chunk in enumerate(split_documents):
#         if i >= max_iterations:
#             break  # Stop after max_iterations
#         executor.submit(process_document_chunk, i, chunk)
#
# print(f"Processing completed in {time.time() - start_time} seconds.")
#
# # Example prompt
# prompt = "process owner name?"
#
# # Generate an embedding for the prompt and retrieve the most relevant doc
# response = ollama.embeddings(
#     prompt=prompt,
#     model="llama3"
# )
#
# results = collection.query(
#     query_embeddings=[response["embedding"]],
#     n_results=1
# )
#
# data = results["documents"][0][0]
#
# # Generate a response combining the prompt and data we retrieved in step 2
# output = ollama.generate(
#     model="llama3",
#     prompt=f"Using this data: {data}. Response to this prompt: {prompt}"
# )
#
# print(output['response'])
