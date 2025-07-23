"""
Test script for Multi-Provider AI Brain System
Tests Google Gemini 1.5 Flash, OpenAI, and local AI integration
"""

import asyncio
import json
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_environment_setup():
    """Test if API keys are properly configured"""
    print("🔧 Testing Environment Setup")
    print("-" * 50)
    
    # Check OpenAI API key
    openai_key = os.getenv('OPENAI_API_KEY')
    if openai_key:
        print(f"✅ OpenAI API Key: {'*' * 20}{openai_key[-10:]}")
    else:
        print("❌ OpenAI API Key: Not configured")
    
    # Check Gemini API key
    gemini_key = os.getenv('GOOGLE_GEMINI_API_KEY')
    if gemini_key and gemini_key != 'your-google-gemini-api-key-here':
        print(f"✅ Gemini API Key: {'*' * 20}{gemini_key[-10:]}")
    else:
        print("❌ Gemini API Key: Not configured (add to .env file)")
    
    print()

def test_imports():
    """Test if all required packages are available"""
    print("📦 Testing Package Imports")
    print("-" * 50)
    
    try:
        import openai
        print("✅ OpenAI package: Available")
    except ImportError:
        print("❌ OpenAI package: Not installed (pip install openai)")
    
    try:
        import google.generativeai as genai
        print("✅ Google Generative AI package: Available")
    except ImportError:
        print("❌ Google Generative AI package: Not installed (pip install google-generativeai)")
    
    try:
        from advanced_ai_brain import advanced_ai_brain
        print("✅ Local AI Brain: Available")
    except ImportError as e:
        print(f"❌ Local AI Brain: Import error - {e}")
    
    print()

async def test_multi_provider_ai():
    """Test the multi-provider AI system"""
    print("🤖 Testing Multi-Provider AI System")
    print("-" * 50)
    
    try:
        from multi_provider_ai_brain import get_multi_provider_response, get_multi_provider_stats
        
        # Test basic response
        test_message = "Hello! Can you help me write a simple Python function to calculate the factorial of a number?"
        
        print(f"Test Message: {test_message}")
        print()
        
        # Test different providers
        providers_to_test = ['local', 'openai', 'gemini']
        
        for provider in providers_to_test:
            print(f"Testing {provider.upper()} provider:")
            try:
                response = await get_multi_provider_response(
                    test_message, 
                    provider=provider,
                    task_type='coding'
                )
                
                print(f"  ✅ Response: {response['response'][:100]}...")
                print(f"  📊 Confidence: {response['confidence']}")
                print(f"  🏷️  Model: {response['model']}")
                print(f"  ⏱️  Response Time: {response['response_time']:.3f}s")
                print()
                
            except Exception as e:
                print(f"  ❌ Error: {e}")
                print()
        
        # Get system stats
        print("📈 System Statistics:")
        stats = get_multi_provider_stats()
        print(json.dumps(stats, indent=2, default=str))
        
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("Make sure all files are in the same directory")

def test_integration_service():
    """Test the Flask integration service"""
    print("🌐 Testing Flask Integration Service")
    print("-" * 50)
    
    try:
        from ai_integration_update import handle_chat_request, handle_stats_request
        
        # Test chat request
        test_message = "What is machine learning?"
        result = handle_chat_request(test_message, task_type='general')
        
        print(f"Chat Request Test:")
        print(f"  ✅ Success: {result['success']}")
        print(f"  📝 Response: {result['response'][:100]}...")
        print(f"  🤖 Provider: {result['provider']}")
        print(f"  📊 Confidence: {result['confidence']}")
        print()
        
        # Test stats request
        stats_result = handle_stats_request()
        print(f"Stats Request Test:")
        print(f"  ✅ Success: {stats_result['success']}")
        print(f"  📊 Providers: {list(stats_result['stats'].get('multi_provider_stats', {}).get('providers', []))}")
        print()
        
    except Exception as e:
        print(f"❌ Integration Service Error: {e}")

def print_setup_instructions():
    """Print setup instructions for missing components"""
    print("📋 Setup Instructions")
    print("-" * 50)
    
    print("1. Install required packages:")
    print("   pip install openai google-generativeai")
    print()
    
    print("2. Get API Keys:")
    print("   - OpenAI: https://platform.openai.com/api-keys")
    print("   - Google Gemini: https://makersuite.google.com/app/apikey")
    print()
    
    print("3. Update .env file:")
    print("   OPENAI_API_KEY=your-openai-api-key-here")
    print("   GOOGLE_GEMINI_API_KEY=your-google-gemini-api-key-here")
    print()
    
    print("4. Run the Flask app:")
    print("   python app.py")
    print()

async def main():
    """Main test function"""
    print("🚀 Multi-Provider AI Brain Test Suite")
    print("=" * 60)
    print()
    
    # Run all tests
    test_environment_setup()
    test_imports()
    await test_multi_provider_ai()
    test_integration_service()
    print_setup_instructions()
    
    print("✨ Test Suite Complete!")

if __name__ == "__main__":
    asyncio.run(main())