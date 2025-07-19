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
    """
    Enriches document chunks with text preprocessing and keyword extraction.
    
    Args:
        documents (list[dict]): List of documents containing 'chunks' field
        
    Returns:
        list[dict]: List of enriched chunk records ready for embedding
        
    This function orchestrates the complete chunk enrichment pipeline:
    1. Processes documents in parallel using multiprocessing for CPU-bound tasks
    2. Enriches each chunk through text normalization, spell correction, stopword removal
    3. Extracts keywords from the combined processed text per document
    4. Converts enriched documents into individual chunk records with metadata
    
    Uses ProcessPoolExecutor for document-level parallelization and ThreadPoolExecutor
    for chunk-level processing within each document for optimal performance.
    
    Environment Variables Required:
        - KEYWORD_EXTRACTION_TOP_N: Number of top keywords to extract per document
    """
    logger.info(f"enrich_chunks() function started - processing {len(documents)} documents")
    logger.info(f"Enriching {len(documents)} documents")
    
    # Determine optimal number of workers
    max_workers = min(len(documents), multiprocessing.cpu_count())
    
    # Process documents in parallel using ProcessPoolExecutor for CPU-bound tasks
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        enriched_docs = list(executor.map(process_single_document, documents))
    
    chunk_records = to_chunk_records(enriched_docs)
    logger.info(f"Enriched {len(documents)} documents into {len(chunk_records)} chunks")
    logger.info(f"enrich_chunks() function completed - created {len(chunk_records)} chunk records")
    return chunk_records


def process_single_document(doc: dict) -> dict:
    """
    Process a single document with parallel chunk processing.
    
    Args:
        doc (dict): Document containing 'chunks' field and metadata
        
    Returns:
        dict: Enriched document with processed chunks and extracted keywords
        
    This function:
    1. Creates a local spell checker instance for this process
    2. Processes all chunks in parallel using ThreadPoolExecutor
    3. Extracts keywords from the combined enriched text
    4. Updates the document with enriched chunks and keywords
    
    Handles empty chunks gracefully and optimizes parallel processing
    based on the number of chunks in the document.
    """
    logger.info(f"process_single_document() function started - document: {doc.get('document_id', 'unknown')}")
    # Create a local spell checker instance for this process
    spell = SpellChecker()
    
    logger.info(f"Enriching document {doc['document_id']}")
    
    # Handle empty chunks case
    if not doc["chunks"]:
        doc["keywords"] = []
        logger.info(f"Document {doc['document_id']} has no chunks to process")
        logger.info(f"process_single_document() function completed - document: {doc.get('document_id', 'unknown')} (no chunks)")
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
    logger.info(f"process_single_document() function completed - document: {doc.get('document_id', 'unknown')}")
    
    return doc


def process_single_chunk(chunk: str, spell: SpellChecker) -> str:
    """
    Process a single chunk through the complete text enrichment pipeline.
    
    Args:
        chunk (str): Raw text chunk to be processed
        spell (SpellChecker): Spell checker instance for correction
        
    Returns:
        str: Enriched and cleaned text chunk
        
    This function applies the following text preprocessing steps in sequence:
    1. Basic text cleaning (lowercase, strip whitespace)
    2. Unicode normalization and non-ASCII character removal
    3. Punctuation removal
    4. Spell correction using the provided spell checker
    5. Stopword removal for noise reduction
    
    The processed text is optimized for embedding generation and keyword extraction.
    """
    logger.info(f"process_single_chunk() function started - chunk length: {len(chunk)} chars")
    cleaned = clean_text(chunk)
    normalized = normalize_unicode(cleaned)
    no_punct = remove_punctuation(normalized)
    no_typos = fix_spelling(no_punct, spell)
    no_stops = remove_stopwords(no_typos)
    logger.info(f"process_single_chunk() function completed - processed chunk length: {len(no_stops)} chars")
    return no_stops


def clean_text(text: str) -> str:
    """
    Performs basic text cleaning operations.
    
    Args:
        text (str): Input text to clean
        
    Returns:
        str: Cleaned text with lowercase and trimmed whitespace
        
    This function applies fundamental text preprocessing:
    - Converts text to lowercase for consistency
    - Strips leading and trailing whitespace
    """
    logger.info(f"clean_text() function started - text length: {len(text)} chars")
    cleaned = text.lower().strip()
    logger.info(f"clean_text() function completed - cleaned text length: {len(cleaned)} chars")
    return cleaned

def normalize_unicode(text: str) -> str:
    """
    Normalizes Unicode text and removes non-ASCII characters.
    
    Args:
        text (str): Input text with potential Unicode issues
        
    Returns:
        str: Normalized text with only ASCII characters
        
    This function:
    1. Applies NFKD Unicode normalization to decompose characters
    2. Removes combining characters (accents, diacritics)
    3. Filters out non-ASCII characters for consistency
    4. Normalizes whitespace to single spaces
    """
    logger.info(f"normalize_unicode() function started - text length: {len(text)} chars")
    text = unicodedata.normalize('NFKD', text)
    text = ''.join(c for c in text if not unicodedata.combining(c))
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    logger.info(f"normalize_unicode() function completed - normalized text length: {len(text)} chars")
    return text

def remove_punctuation(text: str) -> str:
    """
    Removes all punctuation characters from text.
    
    Args:
        text (str): Input text containing punctuation
        
    Returns:
        str: Text with all punctuation characters removed
        
    Uses Python's string.punctuation set to identify and remove
    all standard punctuation marks, leaving only letters, digits, and spaces.
    """
    logger.info(f"remove_punctuation() function started - text length: {len(text)} chars")
    cleaned = text.translate(str.maketrans('', '', string.punctuation))
    logger.info(f"remove_punctuation() function completed - text length: {len(cleaned)} chars")
    return cleaned


def fix_spelling(text: str, spell: SpellChecker) -> str:
    """
    Corrects spelling errors in text using a spell checker.
    
    Args:
        text (str): Input text with potential spelling errors
        spell (SpellChecker): Configured spell checker instance
        
    Returns:
        str: Text with corrected spelling
        
    This function:
    1. Tokenizes the text into individual words
    2. Checks each word against the spell checker dictionary
    3. Applies corrections for misspelled words
    4. Preserves correctly spelled words unchanged
    5. Rejoins corrected words into cleaned text
    """
    logger.info(f"fix_spelling() function started - text length: {len(text)} chars")
    tokens = word_tokenize(text)
    corrected = [spell.correction(word) or word if word not in spell else word for word in tokens]
    result = " ".join(corrected)
    logger.info(f"fix_spelling() function completed - corrected text length: {len(result)} chars")
    return result

def remove_stopwords(text: str) -> str:
    """
    Removes English stopwords from text to reduce noise.
    
    Args:
        text (str): Input text containing stopwords
        
    Returns:
        str: Text with stopwords removed
        
    This function:
    1. Uses NLTK's English stopwords corpus
    2. Tokenizes the text into words
    3. Filters out common stopwords (the, and, is, etc.)
    4. Preserves meaningful content words
    5. Rejoins filtered words into cleaned text
    
    Stopword removal helps focus on important semantic content
    for embedding generation and keyword extraction.
    """
    logger.info(f"remove_stopwords() function started - text length: {len(text)} chars")
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text)
    filtered = [word for word in words if word.lower() not in stop_words]
    result = " ".join(filtered)
    logger.info(f"remove_stopwords() function completed - filtered text length: {len(result)} chars")
    return result

def extract_keywords(text: str, spell: SpellChecker) -> list[str]:
    """
    Extracts the most frequent keywords from processed text.
    
    Args:
        text (str): Preprocessed text for keyword extraction
        spell (SpellChecker): Spell checker instance (unused but kept for consistency)
        
    Returns:
        list[str]: List of top keywords sorted by frequency
        
    This function:
    1. Filters text to alphabetic words only
    2. Removes English stopwords to focus on content
    3. Counts word frequencies using Counter
    4. Returns the top-N most frequent words as keywords
    
    The number of keywords returned is controlled by the
    KEYWORD_EXTRACTION_TOP_N environment variable.
    
    Environment Variables Required:
        - KEYWORD_EXTRACTION_TOP_N: Number of top keywords to extract
    """
    logger.info(f"extract_keywords() function started - text length: {len(text)} chars")
    top_n = int(get_env_var("KEYWORD_EXTRACTION_TOP_N"))
    stop_words = set(stopwords.words('english'))
    words = [w.lower() for w in word_tokenize(text) if w.isalpha() and w.lower() not in stop_words]
    freq = Counter(words)
    keywords = [word for word, count in freq.most_common(top_n)]
    logger.info(f"extract_keywords() function completed - extracted {len(keywords)} keywords")
    return keywords

def to_chunk_records(documents: list[dict]) -> list[dict]:
    """
    Converts enriched documents into individual chunk records for storage.
    
    Args:
        documents (list[dict]): List of enriched documents with chunks and metadata
        
    Returns:
        list[dict]: List of individual chunk records with complete metadata
        
    This function:
    1. Iterates through all documents and their chunks
    2. Creates a separate record for each chunk with:
       - Unique chunk ID
       - Chunk index within the document
       - Original document metadata
       - Processed chunk text
       - Extracted keywords
    3. Preserves all document metadata for each chunk record
    
    The resulting chunk records are ready for embedding generation
    and storage in the vector database.
    """
    logger.info(f"to_chunk_records() function started - processing {len(documents)} documents")
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
    logger.info(f"to_chunk_records() function completed - created {len(chunk_records)} chunk records")
    return chunk_records