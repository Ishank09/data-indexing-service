from data_indexing import utils
from qdrant_client.models import ScoredPoint

def build_prompt_context(query: str, relevant_chunks: list[ScoredPoint]):
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
    return prompt


def render_prompt(context: str, query: str, **kwargs):
    prompt_type = utils.get_env_var("PROMPT_TYPE")
    template = utils.get_env_var(prompt_type)
    return template.format(context=context, query=query, **kwargs)