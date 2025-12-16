import os
import json
from ollama import Client
from generator import load_ollama_config, load_prompts
import sys
from retriever import create_retriever, get_chunks_from_db
from rank_bm25 import BM25Okapi
from runtime_chunker import chunk_row_chunks


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
    
    ollama_config = load_ollama_config()
    client = Client(host=ollama_config["host"])
    
    response = client.generate(model=ollama_config["model"], options={
        "num_ctx": 32768,
        "temperature": 0.0,
        "max_tokens": 256,
        "top_p": 0.9,
        "top_k": 40,
        "frequency_penalty": 0.1,
        "presence_penalty": 0.1,
    }, prompt=prompt)
    
    return response["response"]

def summary_router_chain(query, language, prediction, doc_ids, matched_name):
    query_text = query['query']['content']
    contents = get_contents_from_db(target_doc_ids=doc_ids)
    context = [{"page_content": content} for content in contents]

    if (language == 'en'):
        raw_response = generate_answer(query_text, context, language, prompt_type="new_summary")
    else:
        if (prediction == "Law"):
            raw_response = generate_answer(query_text, context, language, prompt_type="law_summary")
        elif (prediction == "Medical"):
            raw_response = generate_answer(query_text, context, language, prompt_type="medical_summary")
        elif (prediction == "Finance"):
            raw_response = generate_answer(query_text, context, language, prompt_type="finance_summary")
        else:
            raw_response = generate_answer(query_text, context, language, prompt_type="new_summary")
    
    print("raw_response: ", raw_response)
    try:
        # Clean up the response if it contains markdown code blocks
        clean_response = raw_response.replace("```json", "").replace("```", "").strip()
        result_json = json.loads(clean_response)
        answer = result_json.get("answer", '')
        if (not answer): 
            print("JSON Parse answer not found. Retry with fallback prompt")
            answer = generate_answer(query_text, context, language, prompt_type="summary")
    except json.JSONDecodeError:
        print("JSON Parse Error. Retry with fallback prompt")
        answer = generate_answer(query_text, context, language, prompt_type="summary")

    row_chunks = get_chunks_from_db(prediction, doc_ids, language)
    retriever = create_retriever(row_chunks, language)
    big_chunks = retriever.retrieve(answer, threshold=0) # retrieve as much as possible
    
    small_chunks = chunk_row_chunks(big_chunks, language)
    if(not small_chunks):
        return answer, []
    smaller_retriever = create_retriever(small_chunks, language)
    retrieved_chunks = smaller_retriever.retrieve(answer, top1_check=True)
    
    print("Generated Answer:", answer)
    if (retrieved_chunks):
        return answer, retrieved_chunks
    else:
        return answer, []
