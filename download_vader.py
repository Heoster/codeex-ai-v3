#!/usr/bin/env python3
"""
Download NLTK VADER lexicon for sentiment analysis
"""

import nltk
import ssl

# Handle SSL certificate issues
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

def download_vader():
    """Download VADER lexicon"""
    try:
        print("Downloading NLTK VADER lexicon...")
        nltk.download('vader_lexicon')
        print("✅ VADER lexicon downloaded successfully!")
        
        # Test the installation
        from nltk.sentiment import SentimentIntensityAnalyzer
        analyzer = SentimentIntensityAnalyzer()
        
        # Test with sample text
        test_text = "I love this AI assistant! It's amazing and very helpful."
        scores = analyzer.polarity_scores(test_text)
        
        print(f"\n🧪 Testing VADER sentiment analysis:")
        print(f"Text: '{test_text}'")
        print(f"Sentiment scores: {scores}")
        print(f"Overall sentiment: {'Positive' if scores['compound'] > 0.05 else 'Negative' if scores['compound'] < -0.05 else 'Neutral'}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error downloading VADER lexicon: {e}")
        return False

if __name__ == "__main__":
    success = download_vader()
    if success:
        print("\n🎉 VADER lexicon is ready for use in your AI system!")
    else:
        print("\n⚠️ Please try running this script again or check your internet connection.")