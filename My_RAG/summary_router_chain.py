import os
import json
from ollama import Client
from generator import load_ollama_config, load_prompts
import sys
from retriever import create_retriever, get_chunks_from_db
from rank_bm25 import BM25Okapi
from runtime_chunker import chunk_row_chunks
from name_router_chain import create_smaller_chunks_without_names, get_remove_names_from_text


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../db')))
from Connection import Connection
DB_PATH = "db/dataset.db"

conn = Connection(DB_PATH)

def get_contents_from_db(target_doc_ids):
    target_docs = []
    target_set = set(target_doc_ids)

    for id in target_set:
        row = conn.execute("SELECT content FROM documents WHERE doc_id = ?", (id,))
        doc_content = row.fetchone()
        if doc_content:
            content_string = doc_content[0]
            target_docs.append(content_string)
    return target_docs

def generate_answer(query, context_chunks, language="en", prompt_type="summary_chain"):
    context = "\n\n".join([chunk['page_content'] for chunk in context_chunks])
    prompts = load_prompts(type=prompt_type)
    if language not in prompts:
        print(f"Warning: Language '{language}' not found in prompts. Falling back to 'en'.")
        language = "en"

    prompt_template = prompts[language]
    prompt = prompt_template.format(query=query, context=context)
    print("prompt: ", prompt)
    ollama_config = load_ollama_config()
    client = Client(host=ollama_config["host"])
    
    response = client.generate(model=ollama_config["model"], options={
        "num_ctx": 32768,
        "temperature": 0.0,
        # "max_tokens": 256,
        "stop": ["\n\n"],
        # "top_p": 0.9,
        # "top_k": 40,
        # "frequency_penalty": 0.1,
        # "presence_penalty": 0.1,
        # "stream": True,
    }, prompt=prompt)

    print("response: ", response)
    
    return response["response"]

def summary_router_chain(query, language, prediction, doc_ids, matched_name):
    query_text = query['query']['content']
    contents = get_contents_from_db(target_doc_ids=doc_ids)
    # entities = extract_entities(query_text, language)
    modified_query_text = get_remove_names_from_text(query_text, matched_name)
    # context = [{"page_content": content} for content in contents]
    row_chunks = get_chunks_from_db(prediction, doc_ids, language)
    retriever = create_retriever(row_chunks, language)
    big_chunks = retriever.retrieve(modified_query_text, threshold=0) # retrieve as much as possible
    if not big_chunks:
        raw_response = generate_answer(query_text, contents, language, prompt_type="new_summary")
    else:
        if (language == 'en'):
            raw_response = generate_answer(query_text, big_chunks, language, prompt_type="new_summary")
        else:
            if (prediction == "Law"):
                raw_response = generate_answer(query_text, big_chunks, language, prompt_type="law_summary")
            elif (prediction == "Medical"):
                raw_response = generate_answer(query_text, contents, language, prompt_type="medical_summary")
            elif (prediction == "Finance"):
                raw_response = generate_answer(query_text, big_chunks, language, prompt_type="finance_summary")
            else:
                raw_response = generate_answer(query_text, big_chunks, language, prompt_type="new_summary")
    
    print("raw_response: ", raw_response)
    try:
        # Clean up the response if it contains markdown code blocks
        clean_response = raw_response.replace("```json", "").replace("```", "").strip()
        result_json = json.loads(clean_response)
        answer = result_json.get("answer", '')
        if (not answer): 
            print("JSON Parse answer not found. Retry with fallback prompt")
            answer = generate_answer(query_text, big_chunks, language, prompt_type="summary")
    except json.JSONDecodeError:
        print("JSON Parse Error. Retry with fallback prompt")
        answer = generate_answer(query_text, big_chunks, language, prompt_type="summary")

    # row_chunks = get_chunks_from_db(prediction, doc_ids, language)
    # retriever = create_retriever(row_chunks, language)
    # big_chunks = retriever.retrieve(answer, threshold=0) # retrieve as much as possible
    
    small_retrieved_chunks, small_chunks = create_smaller_chunks_without_names(language, big_chunks, matched_name)
    if (not small_chunks):
        return answer, big_chunks
    retriever_2 = create_retriever(small_retrieved_chunks, language)
    retrieved_small_chunks = retriever_2.retrieve(modified_query_text, top1_check=True) # retrieve for higher than the top 1 score * 0.5

    return_chunks = []
    for index, chunk in enumerate(retrieved_small_chunks):
        return_chunks.append(small_chunks[chunk['chunk_index']])
    
    print("Generated Answer:", answer)
    if (return_chunks):
        return answer, return_chunks
    else:
        return answer, []


def extract_entities(query_text, language="en"):
    prompt="""
    Read the query and extract the key **subjects (entities or topics)**.

    ### Constraints
    1. **Use the same language as the query.**
    2. **Extract exact keywords**, do not summarize.
    3. **Format:** Output the subjects separated by commas.
    4. Do not include any additional text or explanation.

    ### Input Query
    {query}

    ### Output
    ###Output###
    """
    ollama_config = load_ollama_config()
    client = Client(host=ollama_config["host"])
    response = client.generate(model=ollama_config["model"], options={
        # "num_ctx": 32768,
        "temperature": 0.0,
        # "max_tokens": 128,
        "stop": ["\n\n"],
        # "top_p": 0.9,
        # "top_k": 40,
        # "frequency_penalty": 0.1,
        # "presence_penalty": 0.1,
        # "stream": True,
    }, prompt=prompt)
    return response["response"]