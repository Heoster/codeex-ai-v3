#!/usr/bin/env python3
"""
🧪 Test script for Enhanced Human-Like AI Brain
"""

from enhanced_human_ai_brain import get_ai_response

def test_enhanced_ai():
    """Test the enhanced AI brain capabilities"""
    
    test_cases = [
        "calculate 25 + 37 * 2",
        "i have went to store yesterday",  # Grammar error
        "what is area of circle with radius 5?",
        "help me with math problems", 
        "how are you today?",
        "solve 2*x**2 + 3*x - 1 = 0",
        "what is 15% of 200?"
    ]
    
    print("🧠 Testing Enhanced Human-Like AI Brain")
    print("=" * 60)
    
    for i, test_input in enumerate(test_cases, 1):
        print(f"\n{i}. User: {test_input}")
        
        try:
            result = get_ai_response(test_input)
            
            print(f"   AI: {result['response']}")
            print(f"   Intent: {result['intent']}")
            print(f"   Sentiment: {result['sentiment']}")
            
            if result['corrected_input'] != test_input:
                print(f"   ✅ Grammar Corrected: '{result['corrected_input']}'")
            
            print(f"   Confidence: {result['confidence']}")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
        
        print("-" * 50)

if __name__ == "__main__":
    test_enhanced_ai()