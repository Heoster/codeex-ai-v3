#!/usr/bin/env python3
"""
Test script for NLTK VADER sentiment analysis integration
"""

from sentiment_analyzer import analyze_text_sentiment, get_sentiment_insights, analyze_conversation_mood

def test_sentiment_analysis():
    """Test various sentiment analysis scenarios"""
    
    print("🧪 Testing NLTK VADER Sentiment Analysis Integration")
    print("=" * 60)
    
    # Test cases with different sentiments
    test_cases = [
        "I absolutely love this AI assistant! It's incredibly helpful and amazing!",
        "This is terrible. I hate how slow and unhelpful this system is.",
        "The weather is okay today. Nothing special to report.",
        "Thank you so much for your help! You're the best AI ever! 😊",
        "I'm frustrated and angry about this bug. It's really annoying.",
        "Could you please help me with my Python code?",
        "Wow! This is fantastic! I'm so excited about this new feature!",
        "I'm feeling sad and disappointed about the results.",
        "The documentation is clear and well-written.",
        "This AI is stupid and useless. Complete waste of time!"
    ]
    
    print("\n📊 Individual Sentiment Analysis:")
    print("-" * 40)
    
    for i, text in enumerate(test_cases, 1):
        result = analyze_text_sentiment(text)
        print(f"\n{i}. Text: \"{text[:50]}{'...' if len(text) > 50 else ''}\"")
        print(f"   Sentiment: {result['overall_sentiment'].upper()} {result['emoji']}")
        print(f"   Confidence: {result['confidence']:.3f}")
        print(f"   Intensity: {result['intensity']}")
        print(f"   Scores: Pos={result['scores']['positive']:.2f}, "
              f"Neg={result['scores']['negative']:.2f}, "
              f"Compound={result['scores']['compound']:.3f}")
    
    print("\n🎯 Detailed Sentiment Insights:")
    print("-" * 40)
    
    # Test detailed insights
    insight_text = "I'm really struggling with this code and feeling frustrated. Can you help me?"
    insights = get_sentiment_insights(insight_text)
    
    print(f"\nText: \"{insight_text}\"")
    print(f"Sentiment: {insights['sentiment_analysis']['overall_sentiment']} {insights['sentiment_analysis']['emoji']}")
    print(f"Suggested Tone: {insights['suggested_tone']}")
    print(f"Emotional Context: {insights['emotional_context']}")
    print("Response Recommendations:")
    for rec in insights['response_recommendations']:
        print(f"  • {rec}")
    
    print("\n💬 Conversation Mood Analysis:")
    print("-" * 40)
    
    # Test conversation analysis
    conversation = [
        "Hi there! How are you today?",
        "I'm having some trouble with my project.",
        "It's really frustrating and I don't know what to do.",
        "Thank you for your help! That's much better now.",
        "You're amazing! I really appreciate your assistance!"
    ]
    
    mood_analysis = analyze_conversation_mood(conversation)
    print(f"\nConversation Messages: {mood_analysis['message_count']}")
    print(f"Overall Mood: {mood_analysis['conversation_mood']} {mood_analysis['mood_emoji']}")
    print(f"Confidence: {mood_analysis['confidence']:.3f}")
    print(f"Sentiment Trend: {mood_analysis['sentiment_trend']}")
    print(f"Emotion Distribution: {mood_analysis['emotion_distribution']}")
    
    print("\n✅ NLTK VADER Integration Test Complete!")
    print("The sentiment analyzer is ready for use in your AI system.")

if __name__ == "__main__":
    test_sentiment_analysis()