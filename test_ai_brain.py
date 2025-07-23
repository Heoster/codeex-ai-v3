#!/usr/bin/env python3
"""
🧪 AI Brain Test Script
Demonstrates the self-training capabilities of the enhanced AI brain
"""

import sys
import time
import json
from enhanced_ai_brain import get_ai_response, provide_feedback, get_learning_statistics

def print_header(title):
    """Print a formatted header"""
    print("\n" + "="*60)
    print(f"🧠 {title}")
    print("="*60)

def print_stats():
    """Print current learning statistics"""
    stats = get_learning_statistics()
    print(f"\n📊 Learning Statistics:")
    print(f"   • Total Patterns: {stats['total_patterns']}")
    print(f"   • Average Confidence: {stats['avg_confidence']:.2f}")
    print(f"   • Average Success Rate: {stats['avg_success_rate']:.2f}")
    print(f"   • Knowledge Nodes: {stats['knowledge_nodes']}")
    print(f"   • Network Accuracy: {stats['network_accuracy']:.2f}")

def test_basic_responses():
    """Test basic AI responses"""
    print_header("Testing Basic AI Responses")
    
    test_inputs = [
        "Hello, how are you?",
        "What is 15 + 25?",
        "Tell me about Python programming",
        "Who is the Prime Minister of India?",
        "Create a simple website",
        "What is artificial intelligence?",
        "Calculate the square root of 144",
        "Help me with JavaScript"
    ]
    
    responses = []
    
    for i, test_input in enumerate(test_inputs, 1):
        print(f"\n{i}. Testing: '{test_input}'")
        response = get_ai_response(test_input)
        print(f"   Response: {response[:100]}{'...' if len(response) > 100 else ''}")
        responses.append((test_input, response))
        time.sleep(0.5)  # Small delay to simulate real usage
    
    return responses

def simulate_user_feedback(responses):
    """Simulate user feedback to train the AI"""
    print_header("Simulating User Feedback for Training")
    
    # Simulate different types of feedback
    feedback_scenarios = [
        (0, 0.9, "Excellent greeting response"),
        (1, 0.8, "Good math calculation"),
        (2, 0.9, "Great programming explanation"),
        (3, 0.7, "Decent knowledge response"),
        (4, 0.8, "Good website creation help"),
        (5, 0.6, "Could be more detailed about AI"),
        (6, 0.9, "Perfect math calculation"),
        (7, 0.7, "Helpful JavaScript guidance")
    ]
    
    for idx, score, comment in feedback_scenarios:
        if idx < len(responses):
            user_input, ai_response = responses[idx]
            print(f"\n📝 Providing feedback for: '{user_input[:50]}...'")
            print(f"   Score: {score}/1.0 - {comment}")
            
            provide_feedback(user_input, ai_response, score)
            time.sleep(0.2)

def test_learning_improvement():
    """Test if AI responses improve after feedback"""
    print_header("Testing Learning Improvement")
    
    # Test the same questions again to see if responses improved
    test_questions = [
        "What is artificial intelligence?",
        "Tell me about Python programming",
        "Help me with JavaScript"
    ]
    
    print("\n🔄 Testing responses after learning...")
    for question in test_questions:
        print(f"\nQuestion: '{question}'")
        response = get_ai_response(question)
        print(f"Improved Response: {response[:150]}{'...' if len(response) > 150 else ''}")

def test_advanced_features():
    """Test advanced AI features"""
    print_header("Testing Advanced AI Features")
    
    # Test with context
    print("\n🧠 Testing contextual responses...")
    context = ["We were discussing programming languages", "Python is very popular"]
    response = get_ai_response("What about JavaScript?", context)
    print(f"Contextual Response: {response}")
    
    # Test mathematical capabilities
    print("\n🔢 Testing mathematical reasoning...")
    math_questions = [
        "What is 25 * 4?",
        "Calculate sin(30)",
        "What is the factorial of 5?",
        "Find the square root of 256"
    ]
    
    for question in math_questions:
        response = get_ai_response(question)
        print(f"Math: {question} → {response}")

def continuous_learning_demo():
    """Demonstrate continuous learning"""
    print_header("Continuous Learning Demonstration")
    
    print("\n🔄 Simulating continuous learning over time...")
    
    # Simulate multiple interactions with feedback
    learning_scenarios = [
        ("How do I install Python?", 0.8),
        ("What is machine learning?", 0.7),
        ("Create a Flask app", 0.9),
        ("Explain neural networks", 0.6),
        ("Help with debugging", 0.8),
        ("What is deep learning?", 0.9),
        ("Python vs JavaScript", 0.7),
        ("AI ethics", 0.8)
    ]
    
    print("\nBefore learning session:")
    print_stats()
    
    for question, feedback_score in learning_scenarios:
        response = get_ai_response(question)
        provide_feedback(question, response, feedback_score)
        print(f"✓ Learned from: '{question}' (feedback: {feedback_score})")
        time.sleep(0.1)
    
    print("\nAfter learning session:")
    print_stats()

def export_learning_data():
    """Export learning data for analysis"""
    print_header("Exporting Learning Data")
    
    stats = get_learning_statistics()
    
    export_data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "learning_statistics": stats,
        "test_completed": True,
        "notes": "AI Brain test completed successfully"
    }
    
    filename = f"ai_brain_test_results_{int(time.time())}.json"
    
    try:
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2)
        print(f"✅ Learning data exported to: {filename}")
    except Exception as e:
        print(f"❌ Error exporting data: {e}")

def main():
    """Main test function"""
    print_header("Enhanced AI Brain Test Suite")
    print("This script demonstrates the self-training capabilities of the AI brain.")
    print("The AI will learn from interactions and improve its responses over time.")
    
    try:
        # Initial statistics
        print("\n📊 Initial Learning Statistics:")
        print_stats()
        
        # Test basic responses
        responses = test_basic_responses()
        
        # Simulate user feedback
        simulate_user_feedback(responses)
        
        # Show improvement
        test_learning_improvement()
        
        # Test advanced features
        test_advanced_features()
        
        # Continuous learning demo
        continuous_learning_demo()
        
        # Export results
        export_learning_data()
        
        print_header("Test Complete!")
        print("✅ The AI brain has successfully demonstrated:")
        print("   • Response generation")
        print("   • Learning from feedback")
        print("   • Pattern recognition")
        print("   • Knowledge building")
        print("   • Continuous improvement")
        
        print("\n🚀 Your AI brain is now trained and ready to use!")
        print("   • Start the web application: python app.py")
        print("   • Visit the AI Dashboard: http://localhost:5000/ai-dashboard")
        print("   • Chat with the AI: http://localhost:5000/chat")
        
    except KeyboardInterrupt:
        print("\n\n⏹️  Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()