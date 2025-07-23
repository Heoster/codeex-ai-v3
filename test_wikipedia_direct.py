#!/usr/bin/env python3
"""
Direct test of Wikipedia integration to debug issues
"""

from wikipedia_knowledge import wikipedia_knowledge

def test_wikipedia_direct():
    """Test Wikipedia integration directly"""
    
    test_queries = [
        "Python programming",
        "artificial intelligence", 
        "solar system",
        "neural networks",
        "machine learning",
        "capital of France"
    ]
    
    print("🔍 Testing Wikipedia API directly")
    print("=" * 50)
    
    for query in test_queries:
        print(f"\n🔎 Testing: '{query}'")
        print("-" * 30)
        
        try:
            # Test search first
            search_results = wikipedia_knowledge.search_wikipedia(query, 3)
            print(f"📊 Search results: {len(search_results)} found")
            
            if search_results:
                print(f"📝 Top result: {search_results[0]['title']}")
                
                # Test getting full knowledge
                knowledge = wikipedia_knowledge.get_knowledge_for_query(query)
                print(f"✅ Knowledge found: {knowledge.get('found', False)}")
                
                if knowledge.get('found'):
                    main_article = knowledge.get('main_article', {})
                    print(f"📚 Article title: {main_article.get('title', 'N/A')}")
                    print(f"📄 Extract length: {len(main_article.get('extract', ''))}")
                else:
                    print("❌ No knowledge retrieved")
            else:
                print("❌ No search results")
                
        except Exception as e:
            print(f"💥 Error: {e}")
    
    print("\n" + "=" * 50)
    print("🏁 Direct Wikipedia test complete!")

if __name__ == "__main__":
    test_wikipedia_direct()