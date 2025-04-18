import ollama
import chromadb

chroma_client = chromadb.Client()
collection = chroma_client.create_collection("RestaurantCollection")

paths = ["WestEnd.txt", "Dietrick.txt"]

def get_text(path):

    with open(path, 'r', encoding='utf-8') as file:
        return file.read() # file.read() returns all content in the file as a String 


documents = []
ids = []

id = 0
for file_path in paths:

    doc_text = get_text(file_path)
    documents.append(doc_text)
    ids.append(f"doc_{id}")

    id += 1

collection.add(
    documents=documents,
    ids=ids
)


query = input("What are your nutrition goals for today?")

closestPages = collection.query(
    query_texts=[query],
    n_results=3
)

response = ollama.chat(
    model="llama3.2",
    messages=[
        {
            "role": "system",
            "content": closestPages["documents"][0][0]
        },
        {
            "role": "system",
            "content": closestPages["documents"][0][1]
        },
        {
            "role": "user",
            "content": query
        }
    ]
)

print(response["message"]["content"])