from qdrant_client import QdrantClient
from FlagEmbedding import FlagModel
from data_indexing import utils
from data_indexing.embedder import load_embedder
import logging

logger = logging.getLogger(__name__)

def retrive_relevant_chunks(query: str):
    """
    Retrieves the most relevant document chunks for a given query.
    
    Args:
        query (str): User query text to find relevant chunks for
        
    Returns:
        list: List of search results containing relevant chunks with scores and metadata
        
    This function implements semantic search by:
    1. Converting the query to a vector embedding using the configured model
    2. Searching the vector database for chunks with highest similarity scores
    3. Returning the top-K most relevant chunks with their metadata
    
    Uses the configured embedding model and vector database connection.
    The number of returned chunks is controlled by RETRIEVER_TOP_K environment variable.
    
    Environment Variables Required:
        - RETRIEVER_TOP_K: Number of top chunks to retrieve
        - VECTOR_DB_URL: Qdrant vector database connection URL
        - VECTOR_DB_COLLECTION_NAME: Collection name in vector database
        - EMBEDDER_MODEL_NAME: Name of the embedding model to use
    """
    logger.info(f"retrive_relevant_chunks() function started - query: {query[:100]}...")
    top_K = int(utils.get_env_var("RETRIEVER_TOP_K"))
    query_embedding = embed_user_query(query)

    client = QdrantClient( utils.get_env_var("VECTOR_DB_URL"))

    search_result = client.search(
        collection_name=utils.get_env_var("VECTOR_DB_COLLECTION_NAME"),
        query_vector=query_embedding.tolist(),
        limit=top_K,
        with_vectors=False,
        with_payload=True,
    )

    logger.info(f"retrive_relevant_chunks() function completed - found {len(search_result)} relevant chunks")
    return search_result


def embed_user_query(query: str):
    """
    Converts user query text into a vector embedding for similarity search.
    
    Args:
        query (str): User query text to be embedded
        
    Returns:
        numpy.ndarray: Vector embedding representation of the query
        
    This function:
    1. Loads the configured embedding model (same as used for chunk embeddings)
    2. Encodes the query text into a vector representation
    3. Returns the embedding for use in vector similarity search
    
    Uses the same embedding model as chunk processing to ensure consistent
    vector space for accurate similarity matching.
    
    Environment Variables Required:
        - EMBEDDER_MODEL_NAME: Name of the embedding model to use
    """
    logger.info(f"embed_user_query() function started - query length: {len(query)} chars")
    encoder = load_embedder()  # Use cached model from embedder module
    embedding = encoder.encode(query)
    logger.info("embed_user_query() function completed - query embedded")
    return embedding