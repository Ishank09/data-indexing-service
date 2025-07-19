

import requests
from data_indexing import utils
import logging

logger = logging.getLogger(__name__)

def generate_llm_response(prompt: str, stream: bool = False):
    """
    Generate a response from the LLM using a generic LLM API.
    
    Args:
        prompt (str): The prompt to generate a response for
        stream (bool, optional): Whether to stream the response. Defaults to False.
        
    Returns:
        str: Generated response text from the LLM
        
    This function uses a generic LLM API compatible with multiple LLMs.
    Separate python LLM inference libraries can be used to generate a response
    from the LLM for example ollama, llama-cpp, etc.
    
    The function:
    1. Retrieves model configuration from environment variables
    2. Constructs API payload with model, prompt, and stream settings
    3. Makes POST request to configured LLM inference URL
    4. Returns the generated response text
    
    Raises:
        RuntimeError: If the API request fails or returns non-200 status code
    """
    logger.info(f"generate_llm_response() function started - prompt length: {len(prompt)} chars")
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
        logger.error(f"generate_llm_response() function failed - API error: {response.status_code}")
        raise RuntimeError(f"OllamaAPI error: {response.status_code}{response.text}")

    logger.info("generate_llm_response() function completed - response generated")
    return response.json()["response"]