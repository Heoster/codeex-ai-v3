#!/usr/bin/env python3
"""
Test LLM Integration
"""

import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_llm_service():
    """Test LLM service directly"""
    print("🧪 Testing LLM Service...")
    
    try:
        from llm_service import llm_service
        
        # Test model status
        print("\n📊 Model Status:")
        status = llm_service.get_model_status()
        for key, value in status.items():
            print(f"  {key}: {value}")
        
        # Test simple response
        print("\n🔹 Test 1: Simple greeting")
        result = llm_service.generate_response("Hello, how are you?")
        print(f"Response: {result.get('response', 'No response')}")
        print(f"Source: {result.get('source', 'Unknown')}")
        print(f"Confidence: {result.get('confidence', 'Unknown')}")
        
        # Test programming question
        print("\n🔹 Test 2: Programming question")
        result = llm_service.generate_response("What is Python?")
        print(f"Response: {result.get('response', 'No response')}")
        print(f"Source: {result.get('source', 'Unknown')}")
        print(f"Confidence: {result.get('confidence', 'Unknown')}")
        
        print("\n✅ LLM Service tests completed!")
        
    except Exception as e:
        print(f"❌ LLM Service test failed: {e}")
        import traceback
        traceback.print_exc()

def test_ai_service():
    """Test enhanced AI service"""
    print("\n🧪 Testing Enhanced AI Service...")
    
    try:
        from ai_service import ai_service
        
        # Test model status
        print("\n📊 AI Service Model Status:")
        status = ai_service.get_model_status()
        for key, value in status.items():
            print(f"  {key}: {value}")
        
        # Test response generation
        print("\n🔹 Test 1: Simple greeting")
        result = ai_service.generate_response("Hello!")
        print(f"Response: {result.get('response', 'No response')}")
        print(f"Source: {result.get('source', 'Unknown')}")
        print(f"Service: {result.get('service', 'Unknown')}")
        
        # Test backward compatibility
        print("\n🔹 Test 2: Simple string response")
        simple_response = ai_service.get_simple_response("What is machine learning?")
        print(f"Simple Response: {simple_response}")
        
        print("\n✅ AI Service tests completed!")
        
    except Exception as e:
        print(f"❌ AI Service test failed: {e}")
        import traceback
        traceback.print_exc()

def test_brain_integration():
    """Test AI brain integration"""
    print("\n🧪 Testing AI Brain Integration...")
    
    try:
        from ai_brain_integration import get_intelligent_response
        
        # Test general query
        print("\n🔹 Test 1: General conversation")
        result = get_intelligent_response("Tell me something interesting")
        print(f"Response: {result.get('response', 'No response')[:200]}...")
        print(f"Source: {result.get('source', 'Unknown')}")
        print(f"Confidence: {result.get('confidence', 'Unknown')}")
        
        print("\n✅ Brain Integration tests completed!")
        
    except Exception as e:
        print(f"❌ Brain Integration test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("🚀 Starting LLM Integration Tests...")
    
    test_llm_service()
    test_ai_service()
    test_brain_integration()
    
    print("\n🎉 All tests completed!")