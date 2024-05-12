from pydantic import BaseModel
import chromadb
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import os
import json
from datetime import datetime


class IndexerConfig(BaseModel):
    doc_dir: str
    persist_path: str
    collection_name: str
    algo: str = "cosine"
    chunk_size: int = 512
    chunk_overlap: int = 64
    top_n_result: int = 5


def json_insert(new_entry: dict[str, str]) -> None:
    if os.path.exists("data.json"):
        # If file.json exists, load its contents
        with open("data.json", "r") as f:
            data = json.load(f)
    else:
        # If file.json doesn't exist, initialize an empty list
        data = dict()
    data.update(new_entry)
    with open("data.json", "w") as f:
        json.dump(data, f, indent=4)
    print("inserted into json")


def json_delete(filename: str) -> None:
    if os.path.exists("data.json"):
        # If file.json exists, load its contents
        with open("data.json", "r") as f:
            data = json.load(f)
        del data[filename]
    else:
        # If file.json doesn't exist, initialize an empty list
        raise ValueError("does not exist")
    with open("data.json", "w") as f:
        json.dump(data, f, indent=4)
    print("deleted from json")


def get_current_time():
    # Get the current time
    current_time = datetime.now()

    # Format the time in a human-readable format
    formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")

    return formatted_time


class indexer:
    def __init__(self, config: IndexerConfig) -> None:
        client = chromadb.PersistentClient(
            path=config.persist_path
        )  # here ad option to select model
        try:
            self.collection = client.get_collection(name=config.collection_name)
        except Exception as e:
            print(f"creating new collection |{config.collection_name}| due to {e}")
            self.collection = client.create_collection(
                name=config.collection_name, metadata={"hnsw:space": config.algo}
            )  # for now lets just use cosine
        self.doc_dir = config.doc_dir
        self.chunk_size = config.chunk_size
        self.chunk_overlap = config.chunk_overlap
        self.n_results = config.top_n_result

    def split_docs(self, documents) -> Document:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
        )
        chunks = text_splitter.split_documents(documents)
        return chunks

    def insert_file(
        self,
        filename: str,
    ) -> None:
        insert_dict = dict()
        filepath = os.path.join(self.doc_dir, filename)
        loader = UnstructuredFileLoader(filepath)
        docs = loader.load()
        print(docs)
        chunks = self.split_docs(docs)
        for idx, chunk in enumerate(chunks):
            self.collection.add(
                documents=[chunk.page_content],
                metadatas={"filename": filename},
                ids=[f"{filename}_{idx}"],
            )
            # print(f"chunk {idx} inserted which is:\n {chunk}")
        insert_dict[filename] = {"last_modified_time": get_current_time()}
        json_insert(insert_dict)

    def delete_file(self, filename: str) -> None:
        self.collection.delete(where={"filename": filename})
        json_delete(filename)
        print(f"{filename} deleted")

    def query_index(self, query):
        top_chunks = self.collection.query(
            query_texts=[query], n_results=self.n_results
        )
        return top_chunks["documents"][0]
