#!/usr/bin/env python3
"""
🧪 Test Improved Intent Recognition for Heoster Questions
Testing various phrasings and semantic variations
"""

from enhanced_human_ai_brain_fixed import get_ai_response

def test_improved_intent_recognition():
    """Test AI's ability to handle varied Heoster-related questions"""
    
    # Test cases with semantic variations
    test_cases = [
        # Direct Heoster questions
        "Who is Heoster?",
        "What is Heoster?", 
        "Tell me about Heoster",
        "What do you know about Heoster?",
        "Can you explain Heoster?",
        "Give me info about Heoster",
        "Heoster information",
        
        # CodeEx AI questions
        "What is CodeEx?",
        "Tell me about CodeEx AI",
        "CodeEx AI features",
        "What can CodeEx do?",
        "About CodeEx AI",
        
        # Creator/company questions
        "Who made this AI?",
        "Who created you?",
        "What company made this?",
        "Who is your creator?",
        "What company are you from?",
        "Who owns this?",
        "Your company",
        "About your company",
        "Company behind this AI",
        
        # Math questions (should still work)
        "calculate 50 + 25 * 2",
        "what is 15% of 300?",
        
        # General questions
        "How are you today?",
        "Help me with programming"
    ]
    
    print("🧪 Testing Improved Intent Recognition for Heoster Questions")
    print("=" * 70)
    
    heoster_detected = 0
    total_heoster_questions = 16  # First 16 are Heoster-related
    
    for i, test_input in enumerate(test_cases, 1):
        print(f"\n{i}. User: {test_input}")
        
        try:
            result = get_ai_response(test_input)
            
            print(f"   AI: {result['response'][:100]}...")
            print(f"   Intent: {result['intent']} | Confidence: {result['confidence']}")
            
            # Track Heoster detection success
            if i <= total_heoster_questions and result['intent'] == 'heoster':
                heoster_detected += 1
                print(f"   ✅ Heoster Knowledge: DETECTED")
            elif i <= total_heoster_questions:
                print(f"   ❌ Heoster Knowledge: MISSED")
            
            if result['response_type'] == 'heoster_response':
                print(f"   🏢 Response Type: Heoster-specific")
            elif result['response_type'] == 'math_solution':
                print(f"   🧮 Response Type: Math solution")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
        
        print("-" * 60)
    
    # Calculate success rate
    success_rate = (heoster_detected / total_heoster_questions) * 100
    print(f"\n📊 HEOSTER DETECTION SUCCESS RATE: {heoster_detected}/{total_heoster_questions} ({success_rate:.1f}%)")
    
    if success_rate >= 80:
        print("🎉 EXCELLENT: Intent recognition is working well!")
    elif success_rate >= 60:
        print("✅ GOOD: Intent recognition needs minor improvements")
    else:
        print("⚠️ NEEDS WORK: Intent recognition requires significant improvement")

if __name__ == "__main__":
    test_improved_intent_recognition()