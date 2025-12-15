from generator import generate_answer

def test_generation():
    # Test Data
    context_en = [{"page_content": "Company A revenue in 2020 was $10 million."}, {"page_content": "Company B revenue in 2020 was $20 million."}]
    context_zh = [{"page_content": "A公司2020年收入为1000万。"}, {"page_content": "B公司2020年收入为2000万。"}]

    queries = [
        # English
        ("What was Company A's revenue?", context_en, "en", "fact"),
        ("Compare Company A and Company B revenue.", context_en, "en", "comparison"),
        ("Summarize the revenue report.", context_en, "en", "summary"),
        # Chinese
        ("A公司收入是多少？", context_zh, "zh", "fact"),
        ("比较A公司和B公司的收入。", context_zh, "zh", "comparison"),
        ("总结收入报告。", context_zh, "zh", "summary"),
    ]

    print("Testing Generation Prompts:")
    for query, ctx, lang, q_type in queries:
        print(f"\n--- Testing {lang.upper()} [{q_type}] ---")
        print(f"Query: {query}")
        try:
            answer = generate_answer(query, ctx, language=lang, type=q_type)
            print(f"Answer: {answer}")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    test_generation()
