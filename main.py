import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from llama_cpp import Llama

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI()

class Prompt(BaseModel):
    payload: str

class ResponseGeneration:
    def __init__(self, model_path: str = "./models/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf", n_gpu_layers: int = -1): # Make n_gpy_layers = 0 for Mac
        self.model_path = model_path
        self.n_gpu_layers = n_gpu_layers
        self.llm = Llama(
            model_path=self.model_path,
            n_gpu_layers=self.n_gpu_layers,
            n_ctx=2048
        )
        logger.info("Llama model initialized with model_path: %s", self.model_path)

    def __call__(self, prompt: str, max_tokens: int = 256, stop: list[str] = ["Q:", "\n"], echo: bool = True):
        try:
            response = self.llm(prompt, max_tokens=max_tokens, stop=stop, echo=echo)
            logger.info("Response generated for prompt: %s", prompt)
            return response
        except Exception as e:
            logger.error("Error generating response: %s", e)
            raise HTTPException(status_code=500, detail="Error generating response")

llm = ResponseGeneration()

@app.post("/response/")
async def create_response(prompt: Prompt):
    try:
        logger.info("Received request with prompt: %s", prompt.payload)
        response = llm(prompt.payload)
        return response
    except HTTPException as e:
        logger.error("HTTPException: %s", e.detail)
        raise e
    except Exception as e:
        logger.error("Unhandled exception: %s", e)
        raise HTTPException(status_code=500, detail="Internal server error")