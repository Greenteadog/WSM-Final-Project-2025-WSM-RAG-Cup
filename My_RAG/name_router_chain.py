from chunker import chunk_documents
from runtime_chunker import chunk_row_chunks
from retriever import create_retriever, get_chunks_from_db
from generator import generate_answer
from rank_bm25 import BM25Okapi

def name_router_chain(query, language="en", prediction=None, doc_id=[], doc_names=[]):
    # if (len(doc_id) == 1):
    #     return single_doc_chain(query, language, prediction, doc_id, matched_name)
    # else:
    #     return multi_doc_chain(query, language, prediction, doc_id, matched_name)
    query_text = query['query']['content']
    modified_query_text = get_remove_names_from_text(query_text, doc_names)
    print("query_text: ", query_text)
    print("modified_query_text: ", modified_query_text)

    # 1. Retrieve bigger chunks(use BM25)
    row_chunks = get_chunks_from_db(prediction, doc_id, language)
    retriever = create_retriever(row_chunks, language)
    
    print("[1] retrieve with bigger chunks:")
    retrieved_chunks = retriever.retrieve(query_text, threshold=0) # retrieve as much as possible
    print('chunks: ', len(retrieved_chunks))

    # 2. Retrieve smaller chunks(use BM25)
    print("[2] retrieve with smaller chunks and extract document name:")
    small_retrieved_chunks, small_chunks = create_smaller_chunks_without_names(language, retrieved_chunks, doc_names)
    retriever_2 = create_retriever(small_retrieved_chunks, language)
    retrieved_small_chunks = retriever_2.retrieve(modified_query_text, top1_check=True) # retrieve for higher than the top 1 score * 0.5
    return_chunks = []
    for index, chunk in enumerate(retrieved_small_chunks):
        return_chunks.append(small_chunks[chunk['chunk_index']])

    print('chunks: ', len(return_chunks))

    # 3. Generate Answer
    print("[3] generate answer:")
    answer = generate_answer(query['query']['content'], return_chunks, language)
    if ("无法回答" in answer or 'Unable to answer' in answer):
        return answer, return_chunks
    
    #4. Fine-tune retriever
    retrieve_answer = get_remove_names_from_text(answer, doc_names)
    final_retrieve = modified_query_text + " " + retrieve_answer
    print("[4] rerieve for final answer: {}".format(final_retrieve))
    retrieved_small_chunks = retriever_2.retrieve(final_retrieve, top1_check=True) # retrieve for higher than the top 1 score * 0.5
    
    return_chunks = []
    for index, chunk in enumerate(retrieved_small_chunks):
        return_chunks.append(small_chunks[chunk['chunk_index']])
    print('final chunks: ', len(return_chunks))
    return answer, return_chunks

########## Helper Functions ##########

def create_smaller_chunks_without_names(language="en", retrieved_chunks=[], doc_names=[]):
    small_chunks = chunk_row_chunks(retrieved_chunks, language)
    small_retrieved_chunks = []
    for index, chunk in enumerate(small_chunks):
        small_retrieved_chunks.append({
            "page_content": get_remove_names_from_text(chunk['page_content'], doc_names),
            "chunk_index": index
        })
    return small_retrieved_chunks, small_chunks

def get_remove_names_from_text(content, doc_names = []):
    if (doc_names):
        for doc_name in doc_names:
            content = content.replace(doc_name, "")
    return content