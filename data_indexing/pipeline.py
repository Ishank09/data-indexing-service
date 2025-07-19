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




