from data_indexing.retriver import retrive_relevant_chunks
from data_indexing.llm import generate_llm_response
from data_indexing.prompt_assembler import build_prompt_context


def anser_query(query:str):
    relevant_chunks = retrive_relevant_chunks(query)
    prompt = build_prompt_context(query, relevant_chunks)
    response = generate_llm_response(prompt)
    return response