from langchain_text_splitters import CharacterTextSplitter
from data_indexing.utils import get_env_var
import logging

logger = logging.getLogger(__name__)

def chunk_text(documents: list[dict]) -> list[dict]:
    """
    Splits document text content into smaller chunks for processing.
    
    Args:
        documents (list[dict]): List of document dictionaries containing 'content' field
        
    Returns:
        list[dict]: Updated documents with 'chunks' field containing text segments
        
    This function processes all documents by:
    1. Loading chunking parameters from environment variables (chunk size, overlap, encoding)
    2. Creating a text splitter using tiktoken encoding for accurate token counting
    3. Splitting each document's content into overlapping chunks
    4. Adding 'chunks' field to each document with the resulting text segments
    
    Uses CharacterTextSplitter with tiktoken encoding to ensure chunks respect
    token limits for downstream processing by embedding models and LLMs.
    
    Environment Variables Required:
        - TOKENIZATION_CHUNK_SIZE: Maximum tokens per chunk
        - TOKENIZATION_CHUNK_OVERLAP: Number of overlapping tokens between chunks  
        - TOKENIZATION_ENCODING_NAME: Tokenizer encoding name (e.g., 'cl100k_base')
    """
    logger.info(f"chunk_text() function started - processing {len(documents)} documents")
    CHUNK_SIZE = int(get_env_var("TOKENIZATION_CHUNK_SIZE"))
    CHUNK_OVERLAP = int(get_env_var("TOKENIZATION_CHUNK_OVERLAP"))
    ENCODING_NAME = get_env_var("TOKENIZATION_ENCODING_NAME")
    logger.info(f"Chunking text with chunk size {CHUNK_SIZE}, chunk overlap {CHUNK_OVERLAP}, encoding name {ENCODING_NAME}")
    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
        encoding_name=ENCODING_NAME, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )

    number_of_chunks = 0
    for doc in documents:
        doc["chunks"] = text_splitter.split_text(doc["content"])
        number_of_chunks += len(doc["chunks"])

    logger.info(f"Chunked {len(documents)} documents into {number_of_chunks} chunks")
    logger.info("chunk_text() function completed")
    return documents