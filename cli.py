from data_indexing.pipeline import run_indexing_job
import logging


logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)    
def main():
    logging.info("Starting data indexing job...")
    run_indexing_job()
    logging.info("Data indexing job completed.")


if __name__ == "__main__":
    main()