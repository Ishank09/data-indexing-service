import os
from dotenv import load_dotenv

load_dotenv()

def get_env_var(key:str) -> str:
    if key in os.environ:
        return os.environ[key]
    else:
        raise ValueError(f"Environment variable {key} is not set")



