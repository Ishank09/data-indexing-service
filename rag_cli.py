# rag_cli.py
from data_indexing.rag_pipeline import anser_query
import logging
from data_indexing import utils

logger = logging.getLogger(__name__)
def main():
    print("Starting custom RAG LLM server...")
    while True:
        query = input("How may I help you today? (type 'exit' to quit)")
        if query.lower() in ["exit", "quit", "bye"]:
            break
        answer = anser_query(query)
        print(f"\nResponse: \n{answer}\n")

    print("Custom RAG LLM server exited.")

if __name__ == "__main__":
    main()
    