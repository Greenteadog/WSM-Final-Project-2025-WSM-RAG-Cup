from classifier import classify_query

def test_classifier():
    test_cases_zh = [
        ("华夏娱乐有限公司2017年的资产收购金额是多少？", "fact"),
        ("比较2017年华夏娱乐有限公司和2018年顶级购物中心的重大资产收购，哪家公司的资产收购金额更大？", "comparison"),
        ("结合绿源环保有限公司2017年的环境与社会责任报告，总结其在2017年采取的环境责任措施。", "summary"),
        ("结合绿源环保有限公司2017年的公司治理报告，绿源环保有限公司是如何通过信息披露和治理改进计划来增强公司透明度的？", "reasoning"),
        ("你好", "general")
    ]
    
    print("Testing Chinese Classifier:")
    for query, expected in test_cases_zh:
        result = classify_query(query, "zh")
        status = "✓" if result == expected else "✗"
        print(f"[{status}] Query: {query[:30]}... -> Expected: {expected}, Got: {result}")

    test_cases_en = [
        ("What is the revenue of Company A?", "fact"),
        ("Compare Company A and Company B.", "comparison"),
        ("Summarize the report.", "summary"),
        ("How did the strategic shift in 2020 affect the profitability?", "reasoning")
    ]
    print("\nTesting English Classifier:")
    for query, expected in test_cases_en:
        result = classify_query(query, "en")
        status = "✓" if result == expected else "✗"
        print(f"[{status}] Query: {query[:30]}... -> Expected: {expected}, Got: {result}")

if __name__ == "__main__":
    test_classifier()
