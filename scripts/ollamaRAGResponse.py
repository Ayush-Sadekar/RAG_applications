import ollama
import chromadb
import pypdf

chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection("RAGPlayground")

# for opening pdfs and indexing them properly
def upload_pdf(file_path):
    
    with open(file_path, "rb") as file:
        pdf_reader = pypdf.PdfReader(file)

        # need unique id's for your vectors so that ChromaDB knows what to pull from
        id = 0

        for page in pdf_reader.pages:
            collection.add(
                documents = [page.extract_text()],
                ids = [f"{file_path}{id}"]
            )
            id+=1

upload_pdf("ChameleonTechPaper.pdf")
upload_pdf("GemmaTechPaper.pdf")

# user can input any query they'd like
query = input(">>>")

# Get top pages with relevant information
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
            "role": "system",
            "content": closestPages["documents"][0][2]
        },
        {
            "role": "user",
            "content": query
        }
    ]
)

print(response["message"]["content"])