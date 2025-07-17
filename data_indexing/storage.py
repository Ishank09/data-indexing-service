from qdrant_client import QdrantClient, models
from data_indexing import utils

collection_name = utils.get_env_var("VECTOR_DB_COLLECTION_NAME")
VECTOR_DB_EMBEDDING_SIZE = int(utils.get_env_var("VECTOR_DB_EMBEDDING_SIZE"))
url = utils.get_env_var("VECTOR_DB_URL")

def create_collection_if_not_exists() -> QdrantClient:
   
    client = QdrantClient(url)
    
    try:
        client.get_collection(collection_name)
    except Exception:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(size=VECTOR_DB_EMBEDDING_SIZE, distance=models.Distance.COSINE),
        )       
    return client


def upsert_chunks(chunk_records: list[dict]):
    client = create_collection_if_not_exists()
    points = []
    for chunk in chunk_records:
        points.append(
            models.PointStruct(
                id=chunk["chunk_id"],
                vector=chunk["embedding"],
                payload={k: v for k, v in chunk.items() if k not in ("embedding")},
            )
        )
    client.upsert(collection_name=collection_name, points=points)




