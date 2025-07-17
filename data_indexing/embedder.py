from FlagEmbedding import FlagModel

from data_indexing import utils

def load_embedder():
    model_name = utils.get_env_var("EMBEDDER_MODEL_NAME")
    model = FlagModel(model_name)
    return model

def embed_chunks(chunk_records: list[dict]):
    model = load_embedder()
    for chunk in chunk_records:
        text = chunk["chunk_text"]
        embedding = model.encode(text)
        chunk["embedding"] = embedding.tolist()
    return chunk_records



