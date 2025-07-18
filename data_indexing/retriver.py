from qdrant_client import QdrantClient
from FlagEmbedding import FlagModel
from data_indexing import utils


def retrive_relevant_chunks(query: str):
    top_K = int(utils.get_env_var("RETRIEVER_TOP_K"))
    query_embedding = embed_user_query(query)

    client = QdrantClient(host=utils.get_env_var("QDRANT_URL"))

    search_result = client.search(
        collection_name=utils.get_env_var("QDRANT_COLLECTION_NAME"),
        query_vector=query_embedding.tolist(),
        limit=top_K,
        with_vectors=False,
        with_payload=True,
    )

    return search_result



def embed_user_query(query: str):
    encoder = FlagModel(utils.get_env_var("EMBEDDER_MODEL_NAME"))
    return encoder.encode(query)