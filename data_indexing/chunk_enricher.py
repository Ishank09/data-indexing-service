from langchain_text_splitters import CharacterTextSplitter
from data_indexing.utils import get_env_var
import logging
import unicodedata
import re
import string
from spellchecker import SpellChecker
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter
from uuid import uuid4
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import multiprocessing
from functools import partial

logger = logging.getLogger(__name__)

def enrich_chunks(documents: list[dict]) -> list[dict]:
    logger.info(f"Enriching {len(documents)} documents")
    
    # Determine optimal number of workers
    max_workers = min(len(documents), multiprocessing.cpu_count())
    
    # Process documents in parallel using ProcessPoolExecutor for CPU-bound tasks
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        enriched_docs = list(executor.map(process_single_document, documents))
    
    chunk_records = to_chunk_records(enriched_docs)
    logger.info(f"Enriched {len(documents)} documents into {len(chunk_records)} chunks")
    return chunk_records


def process_single_document(doc: dict) -> dict:
    """Process a single document with parallel chunk processing"""
    # Create a local spell checker instance for this process
    spell = SpellChecker()
    
    logger.info(f"Enriching document {doc['document_id']}")
    
    # Handle empty chunks case
    if not doc["chunks"]:
        doc["keywords"] = []
        logger.info(f"Document {doc['document_id']} has no chunks to process")
        return doc
    
    # Process chunks in parallel using ThreadPoolExecutor (I/O bound within CPU-bound task)
    max_chunk_workers = min(max(len(doc["chunks"]), 1), 4)  # Ensure at least 1, max 4
    
    with ThreadPoolExecutor(max_workers=max_chunk_workers) as executor:
        # Create a partial function with the spell checker
        process_func = partial(process_single_chunk, spell=spell)
        enriched_chunks = list(executor.map(process_func, doc["chunks"]))
    
    doc["chunks"] = enriched_chunks
    doc["keywords"] = extract_keywords(" ".join(enriched_chunks), spell)
    logger.info(f"Enriched document {doc['document_id']}")
    
    return doc


def process_single_chunk(chunk: str, spell: SpellChecker) -> str:
    """Process a single chunk through the enrichment pipeline"""
    cleaned = clean_text(chunk)
    normalized = normalize_unicode(cleaned)
    no_punct = remove_punctuation(normalized)
    no_typos = fix_spelling(no_punct, spell)
    no_stops = remove_stopwords(no_typos)
    return no_stops


def clean_text(text: str) -> str:
    return text.lower().strip()

def normalize_unicode(text: str)->str:
    text = unicodedata.normalize('NFKD',text)
    text = ''.join(c for c in text if not unicodedata.combining(c))
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def remove_punctuation(text: str) -> str:
    return text.translate(str.maketrans('', '', string.punctuation))


def fix_spelling(text: str, spell: SpellChecker) -> str:
    tokens = word_tokenize(text)
    corrected = [spell.correction(word) or word if word not in spell else word for word in tokens]
    return " ".join(corrected)

def remove_stopwords(text:str) -> str:
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text)
    filtered = [word for word in words if word.lower() not in stop_words]
    return " ".join(filtered)

def extract_keywords(text:str, spell: SpellChecker) -> list[str]:
    top_n = int(get_env_var("KEYWORD_EXTRACTION_TOP_N"))
    stop_words = set(stopwords.words('english'))
    words = [w.lower() for w in word_tokenize(text) if w.isalpha() and w.lower() not in stop_words]
    freq = Counter(words)
    keywords = [word for word, count in freq.most_common(top_n)]
    return keywords

def to_chunk_records(documents: list[dict]) -> list[dict]:
    logger.info(f"Converting {len(documents)} documents to chunk records")
    chunk_records = []
    for doc in documents:
        for i, chunk_text in enumerate(doc["chunks"]):
            chunk_record = {
                 "chunk_id": str(uuid4()),                         # unique chunk id
                "chunk_index": i,
                "chunk_text": chunk_text,
                "doc_id": doc["document_id"],                     # parent doc reference
                "source": doc.get("source"),
                "type": doc.get("type"),
                "title": doc.get("title"),
                "location": doc.get("location"),
                "created_at": doc.get("created_at"),
                "fetched_at": doc.get("fetched_at"),
                "language": doc.get("language"),
                "keywords": doc.get("keywords", []),
            }
            chunk_records.append(chunk_record)
    logger.info(f"Converted {len(documents)} documents to {len(chunk_records)} chunk records")
    return chunk_records