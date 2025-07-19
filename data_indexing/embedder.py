from FlagEmbedding import FlagModel
import logging
from data_indexing import utils
from tqdm import tqdm

logger = logging.getLogger(__name__)

def load_embedder():
    """
    Loads and initializes the embedding model for text vectorization.
    
    Returns:
        FlagModel: Initialized embedding model ready for text encoding
        
    Retrieves the embedding model name from environment variables and
    loads the corresponding FlagEmbedding model. The model is used to
    convert text chunks into vector embeddings for similarity search.
    
    Raises:
        Exception: If model loading fails or model name is not configured
    """
    logger.info("load_embedder() function started")
    model_name = utils.get_env_var("EMBEDDER_MODEL_NAME")
    model = FlagModel(model_name)
    logger.info(f"load_embedder() function completed - loaded model: {model_name}")
    return model

def embed_chunks(chunk_records: list[dict]):
    """
    Generates vector embeddings for all text chunks in the provided records.
    
    Args:
        chunk_records (list[dict]): List of chunk records containing 'chunk_text' field
        
    Returns:
        list[dict]: Updated chunk records with 'embedding' field added
        
    This function:
    1. Loads the configured embedding model
    2. Iterates through all chunk records with progress tracking
    3. Generates vector embeddings for each chunk's text content
    4. Adds embeddings as 'embedding' field in each record
    
    The embeddings are converted to list format for storage compatibility.
    Progress is displayed using tqdm for long-running operations.
    """
    logger.info(f"embed_chunks() function started - processing {len(chunk_records)} chunks")
    model = load_embedder()
    for chunk in tqdm(chunk_records, desc="Embedding chunks"):
        text = chunk["chunk_text"]
        embedding = model.encode(text)
        chunk["embedding"] = embedding.tolist()
    logger.info(f"embed_chunks() function completed - embedded {len(chunk_records)} chunks")
    return chunk_records



