from flask import Flask, request, render_template_string, jsonify, redirect, url_for
from werkzeug.utils import secure_filename
import os
import time
from concurrent.futures import ThreadPoolExecutor
import ollama
from langchain_community.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import chromadb

# Flask app initialization
app = Flask(__name__)
# Path to the folder containing files
UPLOAD_FOLDER = "data"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ChromaDB initialization
client = chromadb.Client()
collection = client.create_collection(name='docs')


# Determine chunk parameters dynamically
def determine_chunk_parameters(documents, max_chunks=1000):
    total_content = "".join([doc.page_content for doc in documents])
    total_length = len(total_content)
    optimal_chunk_size = max(1000, min(total_length // max_chunks, 3000))
    optimal_chunk_overlap = max(100, optimal_chunk_size // 10)
    return optimal_chunk_size, optimal_chunk_overlap


# Process file and load it into ChromaDB
def process_file(file_path):
    extension = os.path.splitext(file_path)[1].lower()
    if extension == ".pdf":
        loader = PyPDFLoader(file_path)
    elif extension in [".docx", ".doc"]:
        loader = UnstructuredWordDocumentLoader(file_path)
    else:
        raise ValueError(f"Unsupported file format: {extension}")

    documents = loader.load()

    chunk_size, chunk_overlap = determine_chunk_parameters(documents)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ".", ",", "\u200b", "\uff0c", "\u3001", "\uff0e", "\u3002", ""]
    )
    splits = text_splitter.split_documents(documents)

    if chunk_size < 1200:
        max_iterations = 5
    else:
        max_iterations = 9


    # Process and embed each chunk
    max_workers = min(32, (os.cpu_count() or 1) + 4)
    start_time = time.time()
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for i, chunk in enumerate(splits):
            if i > max_iterations:
                break  # Stop after max_iterations
            executor.submit(process_document_chunk, i, chunk)


# Process and embed a document chunk
def process_document_chunk(i, chunk):
    try:
        response = ollama.embeddings(model="llama3.2:latest", prompt=chunk.page_content)
        embedding = response["embedding"]
        collection.add(ids=[str(i)], embeddings=[embedding], documents=[chunk.page_content])
    except Exception as e:
        print(f"Error processing chunk {i}: {e}")


# Home route with embedded HTML
@app.route("/")
def index():
    return render_template_string("""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>AI Document Processor</title>
        <link href="https://fonts.googleapis.com/css2?family=Roboto:wght@400;500;700&display=swap" rel="stylesheet">
        <style>
            body {
                font-family: 'Roboto', sans-serif;
                margin: 0;
                padding: 0;
                background-color: #f4f4f9;
                color: #333;
            }
            header {
                background-color: #4CAF50;
                color: white;
                padding: 1rem 0;
                text-align: center;
                font-size: 1.5rem;
                font-weight: 700;
            }
            main {
                padding: 2rem;
                max-width: 800px;
                margin: 0 auto;
                background: white;
                box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.2);
                border-radius: 8px;
            }
            form {
                margin-bottom: 1.5rem;
            }
            label {
                font-weight: 500;
                display: block;
                margin-bottom: 0.5rem;
            }
            input, textarea, button {
                width: 100%;
                padding: 0.8rem;
                margin-bottom: 1rem;
                border: 1px solid #ccc;
                border-radius: 4px;
                font-size: 1rem;
            }
            button {
                background-color: #4CAF50;
                color: white;
                border: none;
                cursor: pointer;
                transition: background-color 0.3s ease;
            }
            button:hover {
                background-color: #45a049;
            }
            #processing {
                font-weight: 500;
                color: #ff5722;
            }
            #uploadResult, #queryResult {
                margin-top: 1.5rem;
                padding: 1rem;
                border: 1px solid #ddd;
                background-color: #f9f9f9;
                border-radius: 4px;
                display: none;
            }

            /* Spinner CSS */
            .spinner {
                display: none;
                margin: 10px auto;
                border: 4px solid #f3f3f3;
                border-top: 4px solid #4CAF50;
                border-radius: 50%;
                width: 30px;
                height: 30px;
                animation: spin 1s linear infinite;
            }

            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
        </style>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/jquery/3.6.0/jquery.min.js"></script>
    </head>
    <body>
        <header>AI Document Processor</header>
        <main>
            <!-- File Upload Form -->
            <form id="uploadForm" enctype="multipart/form-data">
                <label for="file">Upload a file to process:</label>
                <input type="file" id="file" name="file" required>
                <button type="submit">Upload and Process</button>
            </form>
            <div id="processing" style="display: none;">Processing... Please wait.</div>
            <div id="uploadResult"></div>


            <!-- Query Form -->
            <form id="queryForm" style="display: none;" position: relative;>
            <br>
                <label for="prompt">Ask a question about the document:</label>
                <textarea id="prompt" name="prompt" rows="4" required></textarea>
                <button type="submit">Submit Query</button>
                <div class="spinner" id="querySpinner" style="display: none;"></div>
            </form>
            <div id="queryResult"></div>

        </main>

        <script>
            $(document).ready(function () {
                $("#uploadForm").on("submit", function (e) {
                    e.preventDefault();
                    var formData = new FormData(this);
                    $("#processing").show();
                    $("#uploadResult").hide();

                    $.ajax({
                        url: "/upload",
                        type: "POST",
                        data: formData,
                        processData: false,
                        contentType: false,
                        success: function (response) {
                            $("#processing").hide();
                            $("#uploadResult").show().html("File processed successfully! Time taken: " + response.processing_time);
                            $("#queryForm").show();
                        },
                        error: function () {
                            $("#processing").hide();
                            $("#uploadResult").show().html("An error occurred while processing the file.");
                        }
                    });
                });

                $("#queryForm").on("submit", function (e) {
                    e.preventDefault();
                    var prompt = $("#prompt").val();
                    $("#querySpinner").show(); // Show the spinner
                    $("#queryResult").hide();

                    $.ajax({
                        url: "/query",
                        type: "POST",
                        data: { prompt: prompt },
                        success: function (response) {
                            $("#querySpinner").hide(); // Hide the spinner
                            $("#queryResult").show().html("<strong>Response:</strong><br>" + response.response);
                        },
                        error: function () {
                            $("#querySpinner").hide(); // Hide the spinner
                            $("#queryResult").show().html("An error occurred while querying.");
                        }
                    });
                });
            });
        </script>
    </body>
    </html>
    """)


# File upload route
@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"message": "No file part"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"message": "No selected file"}), 400
    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(file_path)

        # Process the file
        start_time = time.time()
        process_file(file_path)
        processing_time = time.time() - start_time
        return jsonify({"message": "File processed successfully!", "processing_time": f"{processing_time:.2f} seconds"})


# Query route
@app.route("/query", methods=["POST"])
def query():
    prompt = request.form.get("prompt")
    response = ollama.embeddings(prompt=prompt, model="llama3.2:latest")
    results = collection.query(query_embeddings=[response["embedding"]], n_results=1)
    data = results["documents"][0][0]
    output = ollama.generate(model="llama3.2:latest",prompt=f"Using this data: {data}. Respond to this prompt: {prompt}")
    return jsonify({"response": output['response']})


# Run the app
if __name__ == "__main__":
    app.run(debug=True)