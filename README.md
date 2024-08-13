# Llama Inference using llama.cpp
A simple implementation for running llama.cpp python wrapper on a FastAPI server instance for local asynchronous inference.

## Setup
### `llama-cpp-python` Installation Command
- Windows
    ```bash
    pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124
    ```

- Mac (MacOS Version is 11.0 or later)
    ```bash
    pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/metal
    ```
### Downloading the Model
1. Go to https://huggingface.co/bartowski/Meta-Llama-3-8B-Instruct-GGUF
2. Download `Meta-Llama-3-8B-Instruct-Q4_K_M.gguf`

    Alternatively, download directly from here [https://huggingface.co/bartowski/Meta-Llama-3-8B-Instruct-GGUF/blob/main/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf](https://huggingface.co/bartowski/Meta-Llama-3-8B-Instruct-GGUF/blob/main/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf)
3. Save the file in ./models

### Installing Dependencies
```bash
pip install -r requirements.txt
```

## Running the Server
```bash
fastapi dev main.py
```

## Client Side Code
- `response_generation.py`
    ```python
    import requests
    from typing import Dict, Any

    class LlamaClient:
        def __init__(self, base_url: str = "http://localhost:8000"):
            self.base_url = base_url

        def generate_response(self, prompt: str, max_tokens: int = 64, stop: list[str] = ["Q:", "\n"], echo: bool = True) -> Dict[str, Any]:
            url = f"{self.base_url}/response/"
            payload = {
                "payload": prompt
            }
            try:
                response = requests.post(url, json=payload)
                response.raise_for_status()
                return response.json()
            except requests.exceptions.HTTPError as http_err:
                print(f"HTTP error occurred: {http_err}")
                raise
            except Exception as err:
                print(f"Other error occurred: {err}")
                raise
    ```

- `__main__.py` (Caller Function)
    ```python
    from response_generation import LlamaClient

    client = LlamaClient(base_url="http://localhost:8000")
    prompt_text = "What is the capital of India?"
    try:
        response = client.generate_response(prompt_text)
        print("Response:", response)
    except Exception as e:
        print(f"An error occurred: {e}")
    ```