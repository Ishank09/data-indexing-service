from data_indexing.pipeline import run_indexing_job
import logging


logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)

logger = logging.getLogger(__name__)

def main():
    """
    Main entry point for the data indexing CLI application.
    
    Executes the complete data indexing pipeline which includes:
    - Loading documents from MongoDB
    - Chunking text content
    - Enriching chunks with metadata and keywords
    - Generating embeddings for chunks
    - Storing chunks in vector database
    
    Logs progress and completion status throughout the process.
    """
    logger.info("Starting data indexing job...")
    run_indexing_job()
    logger.info("Data indexing job completed.")


if __name__ == "__main__":
    main()