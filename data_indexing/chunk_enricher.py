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

logger = logging.getLogger(__name__)
spell = SpellChecker()

def enrich_chunks(documents: list[dict]) -> list[dict]:
    logger.info(f"Enriching {len(documents)} documents")
    for doc in documents:
        enriched_chunks = []
        for chunk in doc["chunks"]:
            cleaned = clean_text(chunk)
            normalized = normalize_unicode(cleaned)
            no_punct = remove_punctuation(normalized)
            no_typos = fix_spelling(no_punct)
            no_stops = remove_stopwords(no_typos)
            enriched_chunks.append(no_stops)
        doc["chunks"] = enriched_chunks
        doc["keywords"] = extract_keywords(" ".join(enriched_chunks))

    chunk_records = to_chunk_records(documents)
    logger.info(f"Enriched {len(documents)} documents into {len(chunk_records)} chunks")
    return chunk_records


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


def fix_spelling(text: str) -> str:
    tokens = word_tokenize(text)
    corrected = [spell.correction(word) or word if word not in spell else word for word in tokens]
    return " ".join(corrected)

def remove_stopwords(text:str) -> str:
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text)
    filtered = [word for word in words if word.lower() not in stop_words]
    return " ".join(filtered)

def extract_keywords(text:str) -> list[str]:
    top_n = int(get_env_var("KEYWORD_EXTRACTION_TOP_N"))
    stop_words = set(stopwords.words('english'))
    words = [w.lower() for w in word_tokenize(text) if w.isalpha() and w.lower() not in stop_words]
    freq = Counter(words)
    keywords = [word for word, count in freq.most_common(top_n)]
    return keywords

def to_chunk_records(documents: list[dict]) -> list[dict]:
    chunk_records = []
    for doc in documents:
        for i, chunk_text in enumerate(doc["chunks"]):
            chunk_record = {
                 "chunk_id": str(uuid4()),                         # unique chunk id
                "chunk_index": i,
                "chunk_text": chunk_text,
                "doc_id": doc["id"],                              # parent doc reference
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
    return chunk_records