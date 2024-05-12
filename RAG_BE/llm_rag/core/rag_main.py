from llm_rag.core.indexer import indexer, IndexerConfig
from llm_rag.core.llm import LLM_config, LLAMA_CPP
from llm_rag.core.llm import LLMResponse
from pydantic import BaseModel
from langchain.schema import Document
import os


class Rag_config(BaseModel):
    IndexerConfig: IndexerConfig
    LLM_config: LLM_config


class rag:
    def __init__(self, config: Rag_config) -> None:
        Indexer_config = config.IndexerConfig
        LLM_config = config.LLM_config
        self.indexer_instance = indexer(Indexer_config)
        self.llm_instance = LLAMA_CPP(LLM_config)
        self.doc_dir = Indexer_config.doc_dir

    def insert_file(self, contents, filename):
        filepath = os.path.join(self.doc_dir, filename)
        with open(filepath, "wb") as file:
            file.write(contents)
        self.indexer_instance.insert_file(filename)

    def delete_file(self, filename: str):
        filepath = os.path.join(self.doc_dir, filename)
        self.indexer_instance.delete_file(filename)
        os.unlink(filepath)

    def retriever(self, query) -> list[Document]:
        return self.indexer_instance.query_index(query)

    def inference(self, query, chunks) -> LLMResponse:
        context = "\n".join(chunks)
        response = self.llm_instance.completion(context=context, query=query)
        print(response)
        return response

    def query(self, query: str) -> dict[str, str | int | float]:
        chunks = self.retriever(query)
        llm_response = self.inference(query, chunks)
        response_dict = {
            "chunk_list": chunks,
            "answer": llm_response.response,
            "llm_response_time": llm_response.llm_response_time,
            "prompt_tokens": llm_response.prompt_tokens,
            "completion_tokens": llm_response.completion_tokens,
            "total_tokens": llm_response.total_tokens,
        }
        return response_dict
