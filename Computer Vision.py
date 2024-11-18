import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import ollama
import chromadb
import json
import urllib
import tensorflow as tf


# Document path where the image file exists
DATA_PATH = "Images/image you want run computer vision on"

# Load the image
image = Image.open(DATA_PATH)


# Define the transformations (resize, normalize, etc.)
transform = transforms.Compose([
    transforms.Resize(256),  # Resize the image to 256x256
    transforms.CenterCrop(224),  # Crop the center 224x224 pixels
    transforms.ToTensor(),  # Convert the image to a tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize as required by ResNet
])

# Preprocess the image
image = transform(image).unsqueeze(0)  # Add batch dimension


# Load a pre-trained ResNet model
model = models.resnet50(pretrained=True)
model.eval()  # Set model to evaluation mode


# Perform the prediction
with torch.no_grad():
    outputs = model(image)
    _, predicted = outputs.max(1)


# Load the ImageNet labels
url = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
response = urllib.request.urlopen(url)
labels = json.loads(response.read())


# Map the predicted class index to the class label
predicted_class = labels[predicted]

# Create a descriptive prompt
prompt = f"This image contains a {predicted_class}. Please provide a detailed description."


# Generate an embedding using the ollama model
embedding_response = ollama.embeddings(
    prompt=prompt,
    model="llama3.1:8b"
)

embedding = embedding_response["embedding"]

# Initialize ChromaDB collection
client = chromadb.Client()
collection = client.create_collection(name='image_descriptions')

# Add the embedding and description to the collection
collection.add(
    ids=[predicted_class],  # Use the class label as the ID
    embeddings=[embedding],
    documents=[prompt]
)

print(f"Embedding generated and added for: {predicted_class}")

# Example retrieval query
query_prompt = "What is that?"
query_response = ollama.embeddings(
    prompt=query_prompt,
    model="llama3.1:8b"
)

results = collection.query(
    query_embeddings=[query_response["embedding"]],
    n_results=1
)

retrieved_data = results["documents"][0][0]

# Generate a response combining the query and retrieved data
output = ollama.generate(
    model="llama3.1:8b",
    prompt=f"Using this data: {retrieved_data}. Response to this prompt: {query_prompt}"
)

print(output['response'])








# # Read and Decode file
# def read_and_decode(filename, reshape_dims):
#     # 1. Read the file
#     img = tf.io.read_file("Images/Peacock.jpg")
#     # 2. Convert the compressed string to a 3D unit8 tensor.
#     img = tf.image.decode_jpeg(img, channels=3)
#     # 3. Convert 3D unit8 to float in the [0,1] range
#     img = tf.image.convert_image_dtype(img, tf.float32)
#     # 4. Resize the image to desire size if needed
#     return tf.image.resize(img, reshape_dims)

