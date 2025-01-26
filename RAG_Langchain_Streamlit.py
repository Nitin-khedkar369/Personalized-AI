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

# Python libraries to convert speech to text and text to speech
import speech_recognition as sr
import pyttsx3

# Add custom CSS for message alignment
st.markdown("""
    <style>
        [data-testid="stChatMessage"] {
            padding: 1.5rem;
            margin-bottom: 0.5rem;
            max-width: 80%;
        }
        .user-message {
            margin-right: auto !important;
            margin-left: 0 !important;
            background-color: #f0f2f6 !important;
            border-radius: 15px;
            padding: 10px;
            display: flex; /* Ensure horizontal alignment */
            align-items: center; /* Vertically center items */
            gap: 10px; /* Space between icon and text */
            white-space: nowrap; /* Prevent text wrapping */
        }
        .assistant-message {
            margin-left: auto !important;
            margin-right: 0 !important;
            background-color: #e6f3ff !important;
            border-radius: 15px;
            padding: 10px;
            display: inline-block;
        }
    </style>
""", unsafe_allow_html=True)

def add_message(role, content):
    """Add a message to the session state"""
    st.session_state.messages.append({"role": role, "content": content})

def display_message(role, content):
    """Display a message with custom styling"""
    css_class = "user-message" if role == "user" else "assistant-message"
    with st.chat_message(role):
        st.markdown(f'<div class="{css_class}">{content}</div>', unsafe_allow_html=True)


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
        # select an embedding model trust me this will make processing the document 10x faster
        response = ollama.embeddings(model="nomic-embed-text:latest", prompt=chunk.page_content)
        embedding = response["embedding"]
        collection.add(
            ids=[str(i)],
            embeddings=[embedding],
            documents=[chunk.page_content]
        )
    except Exception as e:
        print(f"Error processing chunk {i}: {e}")


# Function to convert speech to text and vice versa
def transcribe_speech():
    # Initialize the recognizer
    r = sr.Recognizer()

    # Function to convert text to
    # speech
    # def SpeakText(command):
    #     # Initialize the engine
    #     engine = pyttsx3.init()
    #     engine.say(command)
    #     engine.runAndWait()

    # Loop infinitely for user to
    # speak
    while (1):

        # Exception handling to handle
        # exceptions at the runtime
        try:

            # use the microphone as source for input.
            with sr.Microphone() as source2:

                # wait for a second to let the recognizer
                # adjust the energy threshold based on
                # the surrounding noise level
                r.adjust_for_ambient_noise(source2, duration=0.2)

                # listens for the user's input
                audio2 = r.listen(source2)

                # Using google to recognize audio
                MyText = r.recognize_google(audio2)
                MyText = MyText.lower()

                with st.chat_message("user"):
                    st.write("You said:", MyText)

                return MyText

        except sr.RequestError as e:
            print("Could not request results;{0}:".format(e))

        except sr.UnknownValueError:
            print("unknown error occurred")

# Streamlit Frontend for the app
def main():
    st.title("AI Document Assistant")

    # Let the user upload a file
    uploaded_file = st.file_uploader("Upload a document", type=["pdf", "docx", "doc"])

    if uploaded_file is not None:
        with st.spinner(f"Processing file: {uploaded_file.name}"):
            # st.write(f"Processing file: {uploaded_file.name}")

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
            st.write(f"Processing completed in {time.time() - start_time:.2f} seconds.")

    # Interactive Q&A with text
    st.header("Ask Questions")

    # Starting the session state
    if "chat_history" not in st.session_state:
        st.session_state.messages = []

    prompt = st.chat_input("Enter your query:", key="chat_input")

    # option = ":material/mic:"
    if st.button("🎤 Speak"):

        st.session_state.messages.append(
            {"role": "user",
             "content": st.session_state["chat_input"]},
        )

        with st.spinner("Processing Voice..."):
            speech_text = transcribe_speech()
            start_time = time.time()
            # Generate an embedding for the prompt
            response = ollama.embeddings(prompt=speech_text, model="nomic-embed-text:latest")

            # Retrieve the most relevant document
            results = collection.query(query_embeddings=[response["embedding"]], n_results=1)

            if results["documents"]:
                data = results["documents"][0][0]
                output = ollama.generate(
                    model="llama3.2:latest",
                    prompt=f"Using this data: {data}. Respond to this prompt: {speech_text}"
                )

                st.write(f"Processing your answer completed in {time.time() - start_time:.2f} seconds.")
                # message = st.chat_message("assistant")
                # message.write(output['response'])
                # session state to capture what assistant said
                st.session_state.messages.append(
                    {"role": "assistant",
                     "content": output['response'], },
                )

                # Display chat history with custom styling
                for message in st.session_state.messages:
                    display_message(message["role"], message["content"])


                # Function to convert text to speech
                def SpeakText(command):
                    # Initialize the engine
                    engine = pyttsx3.init()
                    engine.say(command)
                    engine.runAndWait()

                SpeakText(output['response'])
                # listen = st.pills("Start Listening", option, selection_mode="single")

    if prompt:
        # with st.chat_message("user"):
        #     st.write("You said:", prompt)
        with st.spinner("Processing your prompt..."):
            # session state to capture what User said
            st.session_state.messages.append(
                {"role": "user",
                 "content":st.session_state["chat_input"]},
            )

            response = ollama.embeddings(prompt=prompt, model="nomic-embed-text:latest")

            # Retrieve the most relevant document
            results = collection.query(query_embeddings=[response["embedding"]], n_results=1)

            if results["documents"]:
                data = results["documents"][0][0]
                output = ollama.generate(
                    model="llama3.2:latest",
                    prompt=f"Using this data: {data}. Respond to this prompt: {prompt}"
                )

                # session state to capture what assistant said
                st.session_state.messages.append(
                    {"role": "assistant",
                     "content": output['response'], },
                )

                # with st.chat_message("assistant"):
                #     st.write(output['response'])

            for message in st.session_state.messages:
                display_message(message["role"], message["content"])


if __name__ == "__main__":
    main()