from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import Settings
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core import VectorStoreIndex


# create LLM
llm = OpenAI(model="gpt-4o")

#create embedding model
embed_model = OpenAIEmbedding(model="text-embedding-3-small")

Settings.llm = llm
Settings.embed_model = embed_model

# load the documents
data = SimpleDirectoryReader(input_dir="/work/data/",required_exts=[".docx"]).load_data()

# index documents
index = VectorStoreIndex.from_documents(data)

query_engine = index.as_query_engine(similarity_top_k=3)
 
# generate response to query
response = query_engine.query("What are the common themes of the blogs?")