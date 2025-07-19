from data_indexing import utils
from qdrant_client.models import ScoredPoint
import logging

logger = logging.getLogger(__name__)

def build_prompt_context(query: str, relevant_chunks: list[ScoredPoint]):
    """
    Builds a contextual prompt by combining user query with relevant document chunks.
    
    Args:
        query (str): User's query or question
        relevant_chunks (list[ScoredPoint]): Retrieved chunks with similarity scores and metadata
        
    Returns:
        str: Formatted prompt ready for LLM processing
        
    This function assembles a prompt for the LLM by:
    1. Extracting text content from retrieved chunks
    2. Optionally including metadata (location, source, document ID, creation date)
    3. Respecting maximum context length limits to prevent token overflow
    4. Formatting the combined context with the user query using configured template
    
    The function respects environment configuration for:
    - Whether to include metadata with each chunk
    - Maximum character limit for context to control token usage
    - Prompt template type and formatting
    
    Environment Variables Required:
        - INCLUDE_METADATA: Boolean flag to include chunk metadata
        - MAX_CONTEXT_CHARS: Maximum characters allowed in context
        - PROMPT_TYPE: Template type for prompt formatting
    """
    logger.info(f"build_prompt_context() function started - query: {query[:50]}..., chunks: {len(relevant_chunks)}")
    include_metadata = bool(utils.get_env_var("INCLUDE_METADATA"))
    max_context_chars = int(utils.get_env_var("MAX_CONTEXT_CHARS"))
    context = ""
    for chunk in relevant_chunks:
        payload = chunk.payload
        text = payload.get("chunk_text", "") if payload else ""
        if include_metadata:
            meta = []
            if payload and payload.get("location"):
                meta.append(f"Location: {payload['location']}")
            if payload and payload.get("source"):
                meta.append(f"Source: {payload['source']}")
            if payload and payload.get("doc_id"):
                meta.append(f"Document ID: {payload['doc_id']}")
            if payload and payload.get("created_at"):
                meta.append(f"Created At: {payload['created_at']}")
            snippet = (", ".join(meta) + "\n" if meta else "") + text
        else:
            snippet = text + "\n"
        if len(context) + len(snippet) > max_context_chars:
            break
        context += snippet
    prompt = render_prompt(
        context=context,
        query=query,
    )
    logger.info(f"build_prompt_context() function completed - prompt length: {len(prompt)} chars")
    return prompt


def render_prompt(context: str, query: str, **kwargs):
    """
    Renders the final prompt using a configured template.
    
    Args:
        context (str): Assembled context from relevant document chunks
        query (str): User's original query
        **kwargs: Additional template variables
        
    Returns:
        str: Rendered prompt text ready for LLM processing
        
    This function:
    1. Retrieves the prompt template type from environment configuration
    2. Loads the corresponding template string from environment variables
    3. Formats the template with context, query, and any additional variables
    4. Returns the final prompt ready for LLM submission
    
    The prompt template system allows for different formatting styles
    (e.g., instruction-based, chat-based, few-shot examples) controlled
    through environment configuration.
    
    Environment Variables Required:
        - PROMPT_TYPE: Key identifying which prompt template to use
        - {PROMPT_TYPE}: Template string with placeholders for context and query
    """
    logger.info(f"render_prompt() function started - context length: {len(context)}, query length: {len(query)}")
    prompt_type = utils.get_env_var("PROMPT_TYPE")
    template = utils.get_env_var(prompt_type)
    rendered_prompt = template.format(context=context, query=query, **kwargs)
    logger.info(f"render_prompt() function completed - final prompt length: {len(rendered_prompt)} chars")
    return rendered_prompt