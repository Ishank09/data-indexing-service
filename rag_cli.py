# rag_cli.py
from data_indexing.rag_pipeline import anser_query
import logging
from data_indexing import utils

logger = logging.getLogger(__name__)

def main():
    """
    Main entry point for the RAG (Retrieval-Augmented Generation) CLI application.
    
    Provides an interactive command-line interface for users to query the indexed
    document collection. The function:
    - Starts an interactive query loop
    - Processes user queries using the RAG pipeline
    - Retrieves relevant document chunks and generates responses
    - Continues until user exits with 'exit', 'quit', or 'bye'
    
    Uses the configured LLM and vector database for response generation.
    """
    logger.info("Starting custom RAG LLM server...")
    while True:
        query = input("How may I help you today? (type 'exit' to quit)")
        if query.lower() in ["exit", "quit", "bye"]:
            break
        answer = anser_query(query)
        print(f"\nResponse: \n{answer}\n")

    logger.info("Custom RAG LLM server exited.")

if __name__ == "__main__":
    main()
    