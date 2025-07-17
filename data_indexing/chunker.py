from langchain_text_splitters import CharacterTextSplitter
from data_indexing.utils import get_env_var
import logging

logger = logging.getLogger(__name__)

def chunk_text(documents: list[dict]) -> list[dict]:
    CHUNK_SIZE = int(get_env_var("TOKENIZATION_CHUNK_SIZE"))
    CHUNK_OVERLAP = int(get_env_var("TOKENIZATION_CHUNK_OVERLAP"))
    ENCODING_NAME = get_env_var("TOKENIZATION_ENCODING_NAME")
    logger.info(f"Chunking text with chunk size {CHUNK_SIZE}, chunk overlap {CHUNK_OVERLAP}, encoding name {ENCODING_NAME}")
    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
        encoding_name=ENCODING_NAME, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )

    number_of_chunks = 0
    for doc in documents:
        doc["chunks"] = text_splitter.split_text(doc["content"])
        number_of_chunks += len(doc["chunks"])


    logger.info(f"Chunked {len(documents)} documents into {number_of_chunks} chunks")
    return documents