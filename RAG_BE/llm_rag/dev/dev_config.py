from llm_rag.core.indexer import IndexerConfig
from llm_rag.core.rag_main import Rag_config
from llm_rag.core.llm import LLM_config
import yaml


def main():
    indexer_config_instance = IndexerConfig(
        doc_dir="./target_dir",
        persist_path="./persist",
        collection_name="trial_collection",
        algo="cosine",
        chunk_overlap=64,
        chunk_size=512,
        top_n_result=5,
    )
    llm_config_instance = LLM_config(
        rephraser_path="models/mistral-7b-instruct-v0.2.Q4_K_M.gguf",
        max_tokens=1000,  # max tokens that are allowed to be generated.
        stop=[],  # stopping condition
        n_ctx=4096,  # context length
        n_gpu_layers=30,  # layers put to gpu
        prompt_template="<s>[INST]Given the context below\n{context}\nNarrate the answer to the query without using any prior known information.\nquery:\n{query}\nanswer:",
    )
    rag_config_instance = Rag_config(
        IndexerConfig=indexer_config_instance, LLM_config=llm_config_instance
    )
    rag_config_instance = rag_config_instance.dict()
    yaml_file_path = "./dummy2.yaml"
    with open(yaml_file_path, "w") as file:
        yaml.dump(rag_config_instance, file, default_flow_style=False)


if __name__ == "__main__":
    main()
