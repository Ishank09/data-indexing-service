# data_indexing/mongo_loader.py

from data_indexing.utils import get_env_var
from pymongo import MongoClient
import logging


logger = logging.getLogger(__name__)

def get_document_content() -> list[dict]:
    logger.info("Generating document content from MongoDB...")
    client = get_mongo_client()
    db_name = get_env_var("MONGO_DB_NAME")
    collection_name = get_env_var("MONGO_COLLECTION_NAME")

    documents = load_documents(client, db_name, collection_name)


    return documents

def load_documents(client:MongoClient, db_name:str, collection_name:str):
    logger.info(f"Loading documents from {db_name}.{collection_name}...")
    if not is_database_exists(client, db_name):
        logger.error(f"Database {db_name} does not exist")
        raise ValueError(f"Database {db_name} does not exist")

    db = client[db_name]

    if not is_collection_exists(client, db_name, collection_name):
        logger.error(f"Collection {collection_name} does not exist")
        raise ValueError(f"Collection {collection_name} does not exist")

    collection = db[collection_name]

    documents = list(collection.find())
    logger.info(f"Documents loaded, {len(documents)} documents")
    return documents

def is_database_exists(client:MongoClient, db_name:str) -> bool:
    return db_name in client.list_database_names()

def is_collection_exists(client:MongoClient, db_name:str, collection_name:str) -> bool:
    return collection_name in client[db_name].list_collection_names()

def get_mongo_client():
    logger.info("Getting MongoDB client...")
    mongo_uri = get_env_var("MONGO_URI")
    port = int(get_env_var("MONGO_PORT"))
    client = MongoClient(mongo_uri, port)
    logger.info(f"MongoDB client created, {mongo_uri}:{port}")
    return client
