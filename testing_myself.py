import ollama
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain import hub
import bs4
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import Chroma

# LangSmith
import os
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
os.environ['LANGCHAIN_API_KEY'] = "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"


DATA_PATH = "data/Fund Management Policy Guidelines and Procedure.pdf"


# Documents
question = "What are the Milestones?"
document = "Key stages of the project/s as per the grant’s payment schedule"


import tiktoken

def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

print(num_tokens_from_string(question, "cl100k_base"))




embd = OllamaEmbeddings(model="mistral")
query_result = embd.embed_query(question)
document_result = embd.embed_query(document)
len(query_result)

print(f"Embd:{embd}")
print(f"Query_result: {query_result}")
print(f"Document_result: {document_result}")


import numpy as np

def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    return dot_product / (norm_vec1 * norm_vec2)

similarity = cosine_similarity(query_result, document_result)

print("Cosine Similarity:", similarity)



# Loading the document
def load_documents():
    document_loader = PyPDFLoader(DATA_PATH)
    documents = document_loader.load()
    return documents


docs = load_documents()

# print(docs[3])

# Split
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=80,
)

splits = text_splitter.split_documents(docs)

# for s in splits:
#         print(s)

print("This is last")
# Embed
vectorstore = Chroma.from_documents(documents=splits, embedding=embd)
print(vectorstore)
retriever = vectorstore.as_retriever()
print("may be this is last")
#search_kwargs={"k":1}

print(len(retriever.get_relevant_documents("What are the Milestones?")))

#Prompt
prompt = hub.pull("rlm/rag-prompt")
print(f"prompt is:------------------------------>{prompt}")

# LLM
llm = Ollama(model="mistral")
print(f"LLM model is: {llm}")


# Post Processing
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Chain
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    |StrOutputParser()
)

# Questions

rag_chain.invoke("What are the Milestones?")