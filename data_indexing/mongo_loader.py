# data_indexing/mongo_loader.py

from data_indexing.utils import get_env_var
from pymongo import MongoClient
import logging


logger = logging.getLogger(__name__)

def get_document_content() -> list[dict]:
    """
    Retrieves all documents from the configured MongoDB collection.
    
    Returns:
        list[dict]: List of document dictionaries from MongoDB collection
        
    This function orchestrates the document loading process by:
    1. Establishing MongoDB client connection using environment configuration
    2. Retrieving database and collection names from environment variables
    3. Loading all documents from the specified collection
    4. Returning the complete document set for processing
    
    Environment Variables Required:
        - MONGO_DB_NAME: Name of the MongoDB database
        - MONGO_COLLECTION_NAME: Name of the collection containing documents
        - MONGO_URI: MongoDB connection URI
        - MONGO_PORT: MongoDB connection port
        
    Raises:
        ValueError: If database or collection doesn't exist
        Exception: If MongoDB connection fails
    """
    logger.info("get_document_content() function started")
    logger.info("Generating document content from MongoDB...")
    client = get_mongo_client()
    db_name = get_env_var("MONGO_DB_NAME")
    collection_name = get_env_var("MONGO_COLLECTION_NAME")

    documents = load_documents(client, db_name, collection_name)

    logger.info(f"get_document_content() function completed - retrieved {len(documents)} documents")
    return documents

def load_documents(client: MongoClient, db_name: str, collection_name: str):
    """
    Loads all documents from a specific MongoDB collection.
    
    Args:
        client (MongoClient): Connected MongoDB client instance
        db_name (str): Name of the database to access
        collection_name (str): Name of the collection to load from
        
    Returns:
        list[dict]: List of documents from the collection
        
    This function:
    1. Validates that the specified database exists
    2. Validates that the specified collection exists within the database
    3. Retrieves all documents from the collection
    4. Returns the complete document list
    
    Raises:
        ValueError: If database or collection doesn't exist
    """
    logger.info(f"load_documents() function started - loading from {db_name}.{collection_name}")
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
    logger.info(f"load_documents() function completed - loaded {len(documents)} documents")
    return documents

def is_database_exists(client: MongoClient, db_name: str) -> bool:
    """
    Checks if a database exists in the MongoDB instance.
    
    Args:
        client (MongoClient): Connected MongoDB client instance
        db_name (str): Name of the database to check
        
    Returns:
        bool: True if database exists, False otherwise
        
    This function queries the MongoDB instance for available databases
    and checks if the specified database name is present in the list.
    """
    logger.info(f"is_database_exists() function started - checking database: {db_name}")
    exists = db_name in client.list_database_names()
    logger.info(f"is_database_exists() function completed - database '{db_name}' exists: {exists}")
    return exists

def is_collection_exists(client: MongoClient, db_name: str, collection_name: str) -> bool:
    """
    Checks if a collection exists within a specific database.
    
    Args:
        client (MongoClient): Connected MongoDB client instance
        db_name (str): Name of the database containing the collection
        collection_name (str): Name of the collection to check
        
    Returns:
        bool: True if collection exists in the database, False otherwise
        
    This function queries the specified database for available collections
    and checks if the given collection name is present.
    """
    logger.info(f"is_collection_exists() function started - checking {db_name}.{collection_name}")
    exists = collection_name in client[db_name].list_collection_names()
    logger.info(f"is_collection_exists() function completed - collection '{collection_name}' exists: {exists}")
    return exists

def get_mongo_client():
    """
    Creates and returns a MongoDB client connection.
    
    Returns:
        MongoClient: Connected MongoDB client instance
        
    This function:
    1. Retrieves MongoDB connection parameters from environment variables
    2. Creates a new MongoClient instance with the specified URI and port
    3. Returns the connected client for database operations
    
    Environment Variables Required:
        - MONGO_URI: MongoDB connection URI (e.g., 'mongodb://localhost')
        - MONGO_PORT: MongoDB connection port (e.g., 27017)
        
    Raises:
        Exception: If connection to MongoDB fails
    """
    logger.info("get_mongo_client() function started")
    logger.info("Getting MongoDB client...")
    mongo_uri = get_env_var("MONGO_URI")
    port = int(get_env_var("MONGO_PORT"))
    client = MongoClient(mongo_uri, port)
    logger.info(f"MongoDB client created, {mongo_uri}:{port}")
    logger.info("get_mongo_client() function completed")
    return client
