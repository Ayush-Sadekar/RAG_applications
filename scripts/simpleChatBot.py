from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import Settings
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core import VectorStoreIndex
import os

# Path to the API key file
api_key_file = "api_key.txt"

try:
    # Read the API key from the file
    with open(api_key_file, "r") as file:
        api_key = file.read().strip()
    
    # Set the API key in the environment
    os.environ["OPENAI_API_KEY"] = api_key
    print("API key loaded successfully.")
except FileNotFoundError:
    print(f"Error: '{api_key_file}' not found. Please ensure the file exists.")


# create LLM
llm = OpenAI(model="o3-mini")

#create embedding model
embed_model = OpenAIEmbedding(model="text-embedding-3-small")

Settings.llm = llm
Settings.embed_model = embed_model

# load the documents
data = SimpleDirectoryReader(input_dir="/Users/ayush/Desktop/RAG_applications/articles",required_exts=[".txt"]).load_data()

# index documents
index = VectorStoreIndex.from_documents(data)

query_engine = index.as_query_engine(similarity_top_k=3)

# generate response to query
response = query_engine.query("What are the common themes of the blogs?")
print(response)