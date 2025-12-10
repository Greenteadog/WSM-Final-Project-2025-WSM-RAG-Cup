from retriever import create_retriever
from generator import generate_answer

def llm_router_chain(query, language):
    # 1. Do the query expansion
    new_query = expand_query(query, language, 3)
    # new_query = expand_query_2(query, language, 3)
    print("new_query: ", new_query)
    # 2. Retrieve chunks
    retrieved_chunks = retrieve_chunks(query, language)
    # 3. Generate answerß
    answer = generate_answer(query, retrieved_chunks, language)
    # 4. Return answer and chunks
    return answer, retrieved_chunks

def expand_query(query, language="en", size):
    if language == "zh":
        prompt = f"""你是一位專業的搜尋優化專家。
        請為以下查詢的每個關鍵方面提供額外的資訊，使其更容易找到相關文檔（每個查詢約 {size} 個詞）：
        {query}
        
        請將輸出整理成列表，每行一個，不要包含編號、前言或結尾。"""
    else:
        prompt = f"""You are an expert search optimizer. 
        Please provide additional information for each of the key aspects of the following queries that make it easier to find relevant documents (about {size} words per query):
        {query}
        
        Collect the output in a list, one per line, without numbering, preamble, or conclusion."""
    try:
        client = Client()
        response = client.generate(model="granite4:3b", prompt=prompt, stream=False)
        print("expand query response: ", response)
        expanded_keywords = [line.strip().lstrip('0123456789.)-• ')
                    for line in response.get("response", "").split('\n')
                    if line.strip()]
        return [query] + expanded_keywords
    except Exception as e:
        print(f"Error: {e}")
        return [query]


def expand_query_2(query, language="en", size):
    if language == "zh":
        prompt = f"""你是一個有幫助的問答助手。請回答以下問題：
        {query}
        給在回答前出你的思考過程。"""
    else:
        prompt = f"""You are a helpful Q&A assistant. Answer the following question:
        {query}
        Give the rationale before answering."""
    try:
        client = Client()
        response = client.generate(model="granite4:3b", prompt=prompt, stream=False)
        print("expand query response: ", response)
        return [query] + response.get("response", "").split('\n')
    except Exception as e:
        print(f"Error: {e}")
        return [query]

def retrieve_chunks(query, language="en", doc_ids=[], doc_names=[]):
    row_chunks = chunk_row_chunks(query, language, doc_ids, doc_names)
    retriever = create_retriever(row_chunks, language)
    retrieved_chunks = retriever.retrieve(query, top_k=10)
    return retrieved_chunks

def generate_answer(query, retrieved_chunks, language="en"):
    #  TODO: Add query types
    answer = generate_answer(query, retrieved_chunks, language, type="llm_chain")
    return answer