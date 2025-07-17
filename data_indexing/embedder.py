from FlagEmbedding import FlagModel

from data_indexing import utils
from tqdm import tqdm

def load_embedder():
    model_name = utils.get_env_var("EMBEDDER_MODEL_NAME")
    model = FlagModel(model_name)
    return model

def embed_chunks(chunk_records: list[dict]):
    model = load_embedder()
    for chunk in tqdm(chunk_records, desc="Embedding chunks"):
        text = chunk["chunk_text"]
        embedding = model.encode(text)
        chunk["embedding"] = embedding.tolist()
    return chunk_records



