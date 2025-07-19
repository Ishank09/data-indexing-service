from qdrant_client import QdrantClient, models
from data_indexing import utils
import logging

logger = logging.getLogger(__name__)

collection_name = utils.get_env_var("VECTOR_DB_COLLECTION_NAME")
VECTOR_DB_EMBEDDING_SIZE = int(utils.get_env_var("VECTOR_DB_EMBEDDING_SIZE"))
url = utils.get_env_var("VECTOR_DB_URL")

def create_collection_if_not_exists() -> QdrantClient:
    """
    Creates a vector database collection if it doesn't already exist.
    
    Returns:
        QdrantClient: Connected and configured Qdrant client instance
        
    This function:
    1. Establishes connection to the Qdrant vector database
    2. Checks if the configured collection already exists
    3. If not found, creates a new collection with proper vector configuration
    4. Returns the connected client for further operations
    
    The collection is configured with:
    - Vector size from VECTOR_DB_EMBEDDING_SIZE environment variable
    - Cosine distance metric for similarity calculation
    - Collection name from VECTOR_DB_COLLECTION_NAME environment variable
    
    Environment Variables Required:
        - VECTOR_DB_URL: Qdrant database connection URL
        - VECTOR_DB_COLLECTION_NAME: Name of the collection to create/access
        - VECTOR_DB_EMBEDDING_SIZE: Dimension size of vector embeddings
    """
    logger.info("create_collection_if_not_exists() function started")
    client = QdrantClient(url)
    
    try:
        client.get_collection(collection_name)
        logger.info(f"Collection '{collection_name}' already exists")
    except Exception:
        logger.info(f"Creating new collection '{collection_name}'")
        client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(size=VECTOR_DB_EMBEDDING_SIZE, distance=models.Distance.COSINE),
        )
        logger.info(f"Collection '{collection_name}' created successfully")
    
    logger.info("create_collection_if_not_exists() function completed")
    return client


def upsert_chunks(chunk_records: list[dict]):
    """
    Inserts or updates document chunks in the vector database.
    
    Args:
        chunk_records (list[dict]): List of chunk records containing embeddings and metadata
        
    This function:
    1. Ensures the target collection exists in the vector database
    2. Converts chunk records into Qdrant point structures
    3. Extracts vector embeddings and stores metadata as payload
    4. Performs upsert operation to insert new or update existing chunks
    
    Each chunk record should contain:
    - 'chunk_id': Unique identifier for the chunk
    - 'embedding': Vector embedding as a list of floats
    - Additional fields stored as metadata payload (text, document info, etc.)
    
    The upsert operation allows for idempotent indexing - running the same
    chunks multiple times will update rather than duplicate entries.
    """
    logger.info(f"upsert_chunks() function started - processing {len(chunk_records)} chunks")
    client = create_collection_if_not_exists()
    points = []
    for chunk in chunk_records:
        points.append(
            models.PointStruct(
                id=chunk["chunk_id"],
                vector=chunk["embedding"],
                payload={k: v for k, v in chunk.items() if k not in ("embedding")},
            )
        )
    client.upsert(collection_name=collection_name, points=points)
    logger.info(f"upsert_chunks() function completed - upserted {len(chunk_records)} chunks to '{collection_name}'")




