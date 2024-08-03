# Import the dependent libraries
import docx
import chromadb
import ollama
# Importing libraries to read images
import pytesseract
from PIL import Image
# Importing libraries to PDF and Doc
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import UnstructuredWordDocumentLoader
# Importing libraries to split
from langchain.text_splitter import RecursiveCharacterTextSplitter
from concurrent.futures import ThreadPoolExecutor
import time

# Document path where the file exists
DATA_PATH = "data/Fund Management Policy Guidelines and Procedure.pdf"
# DATA_PATH = "data/monopoly.pdf"
# DATA_PATH = "data/SIC - 01 - 03 - 01 - Manage Inadequate Seed Money .pdf"
# DATA_PATH = "data/CE Process.pdf"
# DATA_PATH = "data/AlBawani Proposal v1.0.docx"
# DATA_PATH = "data/Handwritten Image.png"


# Load the required documents
# To Load PDF
document_loader = PyPDFLoader(DATA_PATH)
documents = document_loader.load()

# To Load Word docs
# document_loader = UnstructuredWordDocumentLoader(DATA_PATH)
# documents = document_loader.load()

# # Specify the path to the Tesseract executable if necessary
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


# # Load the image and extract text using pytesseract
# image = Image.open(DATA_PATH)
# text = pytesseract.image_to_string(image)

# # Create a single document containing the extracted text
# documents = [{"page_content": text}]



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
# Also don't forget to put your llm model downloaded locally
def process_document_chunk(i, chunk):
    try:
        # response = ollama.embeddings(model="llama3", prompt=chunk['page_content'])
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
# prompt = "What is NFPs?"
# prompt = "What is ADSE?"
# prompt = "whats the development time?"
prompt = input("Prompt: ")

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
