#!/usr/bin/env python3
"""
Test script to check if AI response generation is working
"""

import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from ai_brain_integration import get_intelligent_response
    
    print("🧪 Testing AI Response Generation...")
    print("=" * 50)
    
    # Test 1: Simple greeting
    print("\n🔹 Test 1: Simple greeting")
    result = get_intelligent_response("Hello, how are you?")
    print(f"Response: {result.get('response', 'No response')}")
    print(f"Source: {result.get('source', 'Unknown')}")
    print(f"Success: {result.get('success', False)}")
    
    # Test 2: Math question
    print("\n🔹 Test 2: Math question")
    result = get_intelligent_response("What is 2 + 2?")
    print(f"Response: {result.get('response', 'No response')}")
    print(f"Source: {result.get('source', 'Unknown')}")
    
    # Test 3: Knowledge question
    print("\n🔹 Test 3: Knowledge question")
    result = get_intelligent_response("What is the capital of France?")
    print(f"Response: {result.get('response', 'No response')}")
    print(f"Source: {result.get('source', 'Unknown')}")
    
    print("\n✅ AI Response test completed!")
    
except Exception as e:
    print(f"❌ Error testing AI response: {e}")
    import traceback
    traceback.print_exc()