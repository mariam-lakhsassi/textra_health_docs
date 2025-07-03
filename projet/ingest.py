import os
import csv
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.embeddings.ollama import OllamaEmbeddings

document_dir = "./documents"
batch_size = 4

def read_txt(file_path):
    with open(file_path, "r", encoding="ISO-8859-1") as f:
        return f.read()

def read_csv(file_path):
    content = ""
    with open(file_path, "r", encoding="ISO-8859-1") as f:
        reader = csv.reader(f)
        for row in reader:
            content += " | ".join(row) + "\n"
    return content

def load_documents_from_directory(directory_path):
    documents = []
    for file in os.listdir(directory_path):
        full_path = os.path.join(directory_path, file)
        if file.endswith(".txt"):
            content = read_txt(full_path)
        elif file.endswith(".csv"):
            content = read_csv(full_path)
        else:
            continue
        documents.append(Document(page_content=content, metadata={"source": file}))
    return documents

def ingest_into_vector_store(documents, db):
    # Utiliser RecursiveCharacterTextSplitter au lieu de CharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=8000,  
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""],  # Plusieurs séparateurs par ordre de priorité
        length_function=len
    )
    
    # Ajouter une gestion d'erreur pour les documents trop grands
    try:
        doc_splits = text_splitter.split_documents(documents)
        db.add_documents(doc_splits)
        db.persist()
    except Exception as e:
        print(f"Erreur lors du découpage des documents: {e}")
        # Pour les documents problématiques, les diviser plus agressivement
        for doc in documents:
            try:
                # Réduire encore la taille pour les documents problématiques
                smaller_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=4000,
                    chunk_overlap=100,
                    separators=["\n\n", "\n", ". ", " ", ""],
                    length_function=len
                )
                smaller_splits = smaller_splitter.split_documents([doc])
                db.add_documents(smaller_splits)
                db.persist()
                print(f"Document {doc.metadata['source']} traité avec une taille de chunk réduite.")
            except Exception as e2:
                print(f"Impossible de traiter le document {doc.metadata['source']}: {e2}")

def initialize_vector_store():
    db = Chroma(
        persist_directory="./TP_db",
        embedding_function=OllamaEmbeddings(model="mxbai-embed-large:latest"),
        collection_name="rag-chroma"
    )
    return db

def main():
    all_documents = load_documents_from_directory(document_dir)
    if all_documents:
        db = initialize_vector_store()
        for i in range(0, len(all_documents), batch_size):
            batch = all_documents[i:i+batch_size]
            ingest_into_vector_store(batch, db)
        print("✅ Données CSV/TXT ingérées dans Chroma.")
    else:
        print("⚠️ Aucun document CSV ou TXT trouvé.")

#main()