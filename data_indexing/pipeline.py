# data_indexing/pipeline.py

from data_indexing.mongo_loader import get_document_content
import logging
from data_indexing.chunker import chunk_text
from data_indexing.chunk_enricher import enrich_chunks

logger = logging.getLogger(__name__)



def run_indexing_job():
    logger.info("Loading documents from MongoDB...")
    documents  = get_document_content()
    logger.info(f"Loaded {len(documents)} documents")

    logger.info("Chunking documents...")
    documents = chunk_text(documents)
    logger.info(f"Chunked {len(documents)} documents")

    logger.info("Enriching chunks...")
    documents = enrich_chunks(documents)
    logger.info(f"Enriched {len(documents)} documents")


