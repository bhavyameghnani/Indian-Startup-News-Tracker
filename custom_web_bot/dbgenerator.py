import os
import json
from langchain.docstore.document import Document
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from typing import list

# Set path to your JSON folder
json_dir = "data_new/articles"

# Load JSON files and convert to LangChain Documents
def load_documents_from_json(json_dir: str) -> list[Document]:
    documents = []
    for file_name in os.listdir(json_dir):
        if file_name.endswith(".json"):
            with open(os.path.join(json_dir, file_name), "r", encoding="utf-8") as f:
                data = json.load(f)
                metadata = {
                    "url": data.get("url"),
                    "tags": data.get("tags", []),
                    "pos": data.get("pos", "")
                }
                content = data.get("summary", "")
                doc = Document(page_content=content, metadata=metadata)
                documents.append(doc)
    return documents

# Load documents
docs = load_documents_from_json(json_dir)
embedding_model = HuggingFaceEmbeddings(model_name="config/model/embedding_model")
vectorstore = FAISS.from_documents(docs, embedding_model)
vectorstore.save_local("faiss_index")

print(f"Indexed {len(docs)} documents into FAISS.")