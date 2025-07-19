from data_indexing.retriver import retrive_relevant_chunks
from data_indexing.llm import generate_llm_response
from data_indexing.prompt_assembler import build_prompt_context
import logging

logger = logging.getLogger(__name__)

def anser_query(query: str):
    """
    Processes a user query through the RAG (Retrieval-Augmented Generation) pipeline.
    
    Args:
        query (str): The user's question or query to be answered
        
    Returns:
        str: Generated response from the LLM based on retrieved relevant context
        
    This function implements the complete RAG workflow:
    1. Retrieves relevant document chunks from the vector database based on query similarity
    2. Builds a contextual prompt combining the query with retrieved chunks
    3. Generates a response using the configured LLM with the enriched prompt
    
    The function uses configured parameters for retrieval (top-K chunks, embedding model)
    and LLM generation (model name, inference URL, context limits).
    """
    logger.info(f"anser_query() function started - processing query: {query[:100]}...")
    relevant_chunks = retrive_relevant_chunks(query)
    prompt = build_prompt_context(query, relevant_chunks)
    response = generate_llm_response(prompt)
    logger.info("anser_query() function completed - response generated")
    return response