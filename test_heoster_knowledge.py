#!/usr/bin/env python3
"""
🧪 Test Heoster Knowledge in Enhanced AI Brain
"""

from enhanced_human_ai_brain import get_ai_response

def test_heoster_knowledge():
    """Test AI knowledge about Heoster"""
    
    test_cases = [
        "What is Heoster?",
        "Tell me about CodeEx AI", 
        "Who created this AI?",
        "What company made this?",
        "What is Heoster Technologies?",
        "What are the features of CodeEx AI?",
        "What technology does Heoster use?"
    ]
    
    print("🧠 Testing Enhanced AI Knowledge about Heoster")
    print("=" * 60)
    
    for i, test_input in enumerate(test_cases, 1):
        print(f"\n{i}. User: {test_input}")
        
        try:
            result = get_ai_response(test_input)
            
            print(f"   AI: {result['response']}")
            print(f"   Intent: {result['intent']} | Confidence: {result['confidence']}")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
        
        print("-" * 50)

if __name__ == "__main__":
    test_heoster_knowledge()