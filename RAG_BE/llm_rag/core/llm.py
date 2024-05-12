from pydantic import BaseModel
import time


class LLM_config(BaseModel):
    rephraser_path: str
    max_tokens: int = 1000  # max tokens that are allowed to be generated.
    stop: list[str] = []  # stopping condition
    n_ctx: int = 4096  # context length
    n_gpu_layers: int = 30  # layers put to gpu
    prompt_template: str = """<s>[INST]Given the context below\n{context}\nNarrate the answer to the query without using any prior known information.\nquery:\n{query}\nanswer:"""


class LLMResponse(BaseModel):
    response: str
    prompt_tokens: int = (
        0  # This is input tokens i.e. your question + the template and system prompt
    )
    completion_tokens: int = 0  # This is output tokens #NOTE: This is how openai wrote it, thats why I also named it like this. The stupidest way to name these two ngl.
    total_tokens: int = 0
    llm_response_time: float = 0.0


class LLAMA_CPP:
    def __init__(self, config: LLM_config) -> None:
        from llama_cpp import Llama

        self.generation_config = config
        self.llm = Llama(
            model_path=self.generation_config.rephraser_path,
            n_gpu_layers=self.generation_config.n_gpu_layers,
            # seed=1337, # Uncomment to set a specific seed  #NOTE: not sure what this is.
            n_ctx=self.generation_config.n_ctx,  # Uncomment to increase the context window
        )

    def completion(self, context, query):
        prompt = self.generation_config.prompt_template.format(
            context=context, query=query
        )
        s_time = time.time()
        output = self.llm(
            prompt=prompt,
            max_tokens=self.generation_config.max_tokens,
            stop=self.generation_config.stop,
            echo=False,  # This means it wont repeate whatever we ask it.
        )
        e_time = time.time()
        print(output)
        response = LLMResponse(
            response=output["choices"][0]["text"],
            llm_response_time=e_time - s_time,
            prompt_tokens=output["usage"]["prompt_tokens"],
            completion_tokens=output["usage"]["completion_tokens"],
            total_tokens=output["usage"]["total_tokens"],
        )
        return response
