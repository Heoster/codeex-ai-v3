#!/usr/bin/env python3
"""
🧪 Complete Test of Enhanced AI System with Heoster Knowledge
"""

from enhanced_human_ai_brain_fixed import get_ai_response

def test_complete_ai_system():
    """Test all AI capabilities including Heoster knowledge"""
    
    test_cases = [
        # Heoster Knowledge Tests
        "What is Heoster?",
        "Tell me about CodeEx AI features", 
        "Who created this AI?",
        "What company made this?",
        
        # Math Capabilities Tests
        "calculate 25 * 15 + 100 / 4",
        "what is 15% of 200?",
        "solve x^2 + 5x - 6 = 0",
        
        # Grammar Correction Tests
        "i have went to store yesterday",
        "he don't know the answer",
        
        # General Knowledge Tests
        "What is the area of circle with radius 10?",
        "how are you doing today?",
        "help me with programming"
    ]
    
    print("🧠 Testing Complete Enhanced AI System with Heoster Knowledge")
    print("=" * 70)
    
    for i, test_input in enumerate(test_cases, 1):
        print(f"\n{i}. User: {test_input}")
        
        try:
            result = get_ai_response(test_input)
            
            print(f"   AI: {result['response']}")
            print(f"   Intent: {result['intent']} | Confidence: {result['confidence']}")
            
            if result['corrected_input'] != test_input:
                print(f"   ✅ Grammar Fixed: \"{result['corrected_input']}\"")
            
            # Show special features
            if result['response_type'] == 'heoster_response':
                print(f"   🏢 Heoster Knowledge: ACTIVE")
            elif result['response_type'] == 'math_solution':
                print(f"   🧮 Math Engine: ACTIVE")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
        
        print("-" * 60)

if __name__ == "__main__":
    test_complete_ai_system()