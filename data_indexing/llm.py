

import requests
from data_indexing import utils
import logging
logger = logging.getLogger(__name__)
"""
Generate a response from the LLM. This used generic LLM API, which is compatible with multiple LLMs.
Seperate python LLM inference libraries can be used to generate a response from the LLM for example ollama, llama-cpp, etc.


Args:
    prompt: The prompt to generate a response for
    stream: Whether to stream the response
"""
def generate_llm_response(prompt: str, stream:bool = False):
    model_env_var = utils.get_env_var("LLM_MODEL_NAME")
    model = utils.get_env_var(model_env_var)
    logger.info(f"Using model: {model}")
    llm_url = utils.get_env_var("LLM_INFERENCE_URL")
    stream_mode = bool(utils.get_env_var("LLM_STREAM_MODE"))

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": stream,
    }

    response = requests.post(llm_url, json=payload)

    if response.status_code != 200:
        raise RuntimeError(f"OllamaAPI error: {response.status_code}{response.text}")

   
    return response.json()["response"]