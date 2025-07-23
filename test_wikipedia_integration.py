#!/usr/bin/env python3
"""
Test script for Wikipedia integration with "What" questions
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ai_brain_integration import get_intelligent_response

def test_what_questions():
    """Test various "What" questions to verify Wikipedia integration"""
    
    test_questions = [
        "What is Python programming?",
        "What is artificial intelligence?",
        "What is the solar system?",
        "What are neural networks?",
        "What is machine learning?",
        "What is the capital of France?",  # This might not trigger Wikipedia but should work
        "Calculate 2 + 2",  # This should NOT trigger Wikipedia
    ]
    
    print("🧪 Testing Wikipedia Integration for 'What' Questions")
    print("=" * 60)
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n{i}. Testing: '{question}'")
        print("-" * 40)
        
        try:
            response = get_intelligent_response(question)
            
            # Check if Wikipedia was used
            source = response.get('source', 'unknown')
            success = response.get('success', False)
            
            print(f"✅ Source: {source}")
            print(f"✅ Success: {success}")
            
            if source == 'wikipedia':
                print("🎉 Wikipedia integration WORKING!")
                print(f"📚 Response preview: {response['response'][:200]}...")
            elif question.lower().startswith('what'):
                print("⚠️  Wikipedia not triggered (might be fallback)")
                print(f"📝 Response preview: {response['response'][:200]}...")
            else:
                print("✅ Non-Wikipedia question handled correctly")
                print(f"📝 Response preview: {response['response'][:200]}...")
                
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print("\n" + "=" * 60)
    print("🏁 Wikipedia Integration Test Complete!")

if __name__ == "__main__":
    test_what_questions()