import chromadb
import ollama
from langchain.text_splitter import RecursiveCharacterTextSplitter
from concurrent.futures import ThreadPoolExecutor
import time
import pytesseract
from PIL import Image
from langchain_community.document_loaders import UnstructuredImageLoader

# Specify the path to the Tesseract executable if necessary
# Download OCR engine from https://github.com/UB-Mannheim/tesseract/wiki if required or used other OCR engine
# For example, on Windows:
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Document path where the image file exists
DATA_PATH = "OCR/Handwritten Image.jpg"

# Load the image and extract text using pytesseract
image = Image.open(DATA_PATH)
text = pytesseract.image_to_string(image)


# Create a single document containing the extracted text
documents = [{"text": text}]


# Defining the Split i.e. how to split, chunk size, etc.
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
# splits = text_splitter.split_documents(documents)
splits = text_splitter.split_text(text)

# Initialize ChromaDB collection
client = chromadb.Client()
collection = client.create_collection(name='docs')

max_iterations = 5

# Function to process a single document chunk
def process_document_chunk(i, chunk):
    try:
        response = ollama.embeddings(model="llama3", prompt=chunk)
        embedding = response["embedding"]
        collection.add(
            ids=[str(i)],
            embeddings=[embedding],
            documents=[chunk]
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
prompt = "What is the image about?"

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
    model="llama3",
    prompt=f"Using this data: {data}. Response to this prompt: {prompt}"
)

print(output['response'])





























