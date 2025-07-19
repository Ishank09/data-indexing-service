import os
import logging
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

def get_env_var(key: str) -> str:
    """
    Retrieves an environment variable value by key.
    
    Args:
        key (str): The environment variable name to retrieve
    
    Returns:
        str: The value of the environment variable
        
    Raises:
        ValueError: If the environment variable is not set
        
    This function checks if the specified environment variable exists
    and returns its value. If the variable is not found, raises a
    descriptive error message.
    """
    if key in os.environ:
        value = os.environ[key]
        return value
    else:
        logger.error(f"get_env_var() function failed - Environment variable {key} is not set")
        raise ValueError(f"Environment variable {key} is not set")



