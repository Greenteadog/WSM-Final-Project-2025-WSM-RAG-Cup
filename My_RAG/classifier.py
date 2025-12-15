from ollama import Client
from utils import load_ollama_config
from generator import load_prompts

def classify_query(query, language="en"):
    """
    Classifies the query into 'fact', 'comparison', 'summary', or 'general'.
    """
    try:
        prompts = load_prompts("classification")
        if language not in prompts:
            language = "en"
        
        prompt_template = prompts[language]
        prompt = prompt_template.format(query=query)
        
        ollama_config = load_ollama_config()
        client = Client(host=ollama_config["host"])
        
        # Use a smaller model for classification if possible, or the same main model
        response = client.generate(model=ollama_config["model"], options={
            "temperature": 0.0, # Deterministic
            "max_tokens": 10,
        }, prompt=prompt)
        
        category = response["response"].strip().lower()
        
        # Validate category
        valid_categories = ["fact", "comparison", "summary", "reasoning", "general"]
        if category not in valid_categories:
            print(f"[Classifier] Warning: Invalid category '{category}', defaulting to 'general'")
            return "general"
            
        print(f"[Classifier] Query detected as: {category}")
        return category

    except Exception as e:
        print(f"[Classifier] Error: {e}")
        return "general"

if __name__ == "__main__":
    # Test
    queries = [
        "What happened in 2020?",
        "Compare Company A and Company B.",
        "Summarize the main points."
    ]
    for q in queries:
        print(f"Query: {q} -> {classify_query(q, 'en')}")
