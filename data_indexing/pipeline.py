# data_indexing/pipeline.py

from data_indexing.mongo_loader import get_document_content
import logging
from data_indexing.chunker import chunk_text
from data_indexing.chunk_enricher import enrich_chunks
from data_indexing.embedder import embed_chunks
from data_indexing import utils
from data_indexing.storage import upsert_chunks

logger = logging.getLogger(__name__)


def run_indexing_job():
    """
    Executes the complete document indexing pipeline.
    
    This function orchestrates the entire data indexing workflow:
    1. Loads documents from MongoDB using configured connection parameters
    2. Chunks the document text into smaller segments for processing
    3. Enriches chunks with metadata, keywords, and preprocessing
    4. Generates vector embeddings for each chunk using the configured model
    5. Stores enriched chunks with embeddings in the vector database
    
    The pipeline processes documents in sequence with progress logging
    at each step. Uses environment variables for configuration of chunk
    sizes, embedding models, and database connections.
    
    Raises:
        Exception: If any step in the pipeline fails, propagates the error
    """
    logger.info("Starting data indexing job...")

    logger.info("Loading documents from MongoDB...")
    documents  = get_document_content()
    logger.info(f"Loaded {len(documents)} documents")

    logger.info("Chunking documents...")
    documents = chunk_text(documents)
    logger.info(f"Chunked {len(documents)} documents")

    logger.info("Enriching chunks...")
    chunk_records = enrich_chunks(documents)
    logger.info(f"Enriched {len(chunk_records)} chunks")

    logger.info("Embedding chunks...")
    chunk_records = embed_chunks(chunk_records)
    logger.info(f"Embedded {len(chunk_records)} chunks")

    logger.info(f"Saving chunks to {utils.get_env_var('VECTOR_DB_COLLECTION_NAME')}...")
    upsert_chunks(chunk_records)
    logger.info(f"Saved {len(chunk_records)} chunks")

    logger.info("Indexing job completed successfully")




