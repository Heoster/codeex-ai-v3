#!/usr/bin/env python3
"""
Enhanced Sentiment Analysis Service using NLTK VADER
Integrates with CodeEx AI for emotional intelligence
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

logger = logging.getLogger(__name__)

class EnhancedSentimentAnalyzer:
    """Advanced sentiment analysis with VADER lexicon"""
    
    def __init__(self):
        try:
            self.analyzer = SentimentIntensityAnalyzer()
            self.sentiment_history = []
            logger.info("VADER sentiment analyzer initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing sentiment analyzer: {e}")
            self.analyzer = None
    
    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment of text with detailed breakdown"""
        if not self.analyzer:
            return self._fallback_sentiment(text)
        
        try:
            # Get VADER scores
            scores = self.analyzer.polarity_scores(text)
            
            # Determine overall sentiment
            compound = scores['compound']
            if compound >= 0.05:
                overall = 'positive'
                emoji = '😊'
            elif compound <= -0.05:
                overall = 'negative'
                emoji = '😔'
            else:
                overall = 'neutral'
                emoji = '😐'
            
            # Calculate confidence
            confidence = abs(compound)
            
            # Determine intensity
            if abs(compound) >= 0.7:
                intensity = 'strong'
            elif abs(compound) >= 0.3:
                intensity = 'moderate'
            else:
                intensity = 'weak'
            
            result = {
                'text': text,
                'overall_sentiment': overall,
                'confidence': confidence,
                'intensity': intensity,
                'emoji': emoji,
                'scores': {
                    'positive': scores['pos'],
                    'negative': scores['neg'],
                    'neutral': scores['neu'],
                    'compound': scores['compound']
                },
                'timestamp': datetime.now().isoformat(),
                'analysis_method': 'VADER'
            }
            
            # Store in history
            self.sentiment_history.append(result)
            if len(self.sentiment_history) > 100:  # Keep last 100 analyses
                self.sentiment_history.pop(0)
            
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {e}")
            return self._fallback_sentiment(text)
    
    def _fallback_sentiment(self, text: str) -> Dict[str, Any]:
        """Fallback sentiment analysis without VADER"""
        positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 'love', 'like', 'happy', 'pleased']
        negative_words = ['bad', 'terrible', 'awful', 'horrible', 'hate', 'dislike', 'sad', 'angry', 'frustrated', 'disappointed']
        
        text_lower = text.lower()
        pos_count = sum(1 for word in positive_words if word in text_lower)
        neg_count = sum(1 for word in negative_words if word in text_lower)
        
        if pos_count > neg_count:
            overall = 'positive'
            emoji = '😊'
            compound = 0.5
        elif neg_count > pos_count:
            overall = 'negative'
            emoji = '😔'
            compound = -0.5
        else:
            overall = 'neutral'
            emoji = '😐'
            compound = 0.0
        
        return {
            'text': text,
            'overall_sentiment': overall,
            'confidence': abs(compound),
            'intensity': 'moderate',
            'emoji': emoji,
            'scores': {
                'positive': pos_count / max(pos_count + neg_count, 1),
                'negative': neg_count / max(pos_count + neg_count, 1),
                'neutral': 0.5,
                'compound': compound
            },
            'timestamp': datetime.now().isoformat(),
            'analysis_method': 'fallback'
        }
    
    def analyze_conversation_mood(self, messages: List[str]) -> Dict[str, Any]:
        """Analyze overall mood of a conversation"""
        if not messages:
            return {'mood': 'neutral', 'confidence': 0.0}
        
        sentiments = [self.analyze_sentiment(msg) for msg in messages]
        
        # Calculate average sentiment
        avg_compound = sum(s['scores']['compound'] for s in sentiments) / len(sentiments)
        avg_confidence = sum(s['confidence'] for s in sentiments) / len(sentiments)
        
        # Determine conversation mood
        if avg_compound >= 0.1:
            mood = 'positive'
            mood_emoji = '😊'
        elif avg_compound <= -0.1:
            mood = 'negative'
            mood_emoji = '😔'
        else:
            mood = 'neutral'
            mood_emoji = '😐'
        
        # Find dominant emotions
        emotions = [s['overall_sentiment'] for s in sentiments]
        emotion_counts = {
            'positive': emotions.count('positive'),
            'negative': emotions.count('negative'),
            'neutral': emotions.count('neutral')
        }
        
        return {
            'conversation_mood': mood,
            'mood_emoji': mood_emoji,
            'confidence': avg_confidence,
            'average_compound': avg_compound,
            'message_count': len(messages),
            'emotion_distribution': emotion_counts,
            'sentiment_trend': self._calculate_trend(sentiments),
            'timestamp': datetime.now().isoformat()
        }
    
    def _calculate_trend(self, sentiments: List[Dict]) -> str:
        """Calculate sentiment trend over conversation"""
        if len(sentiments) < 3:
            return 'stable'
        
        # Compare first third with last third
        first_third = sentiments[:len(sentiments)//3]
        last_third = sentiments[-len(sentiments)//3:]
        
        first_avg = sum(s['scores']['compound'] for s in first_third) / len(first_third)
        last_avg = sum(s['scores']['compound'] for s in last_third) / len(last_third)
        
        diff = last_avg - first_avg
        
        if diff > 0.1:
            return 'improving'
        elif diff < -0.1:
            return 'declining'
        else:
            return 'stable'
    
    def get_sentiment_insights(self, text: str) -> Dict[str, Any]:
        """Get detailed sentiment insights for AI response adaptation"""
        analysis = self.analyze_sentiment(text)
        
        # Generate response recommendations
        recommendations = []
        
        if analysis['overall_sentiment'] == 'negative':
            recommendations.extend([
                'Use empathetic language',
                'Offer helpful solutions',
                'Acknowledge the user\'s feelings',
                'Be supportive and understanding'
            ])
        elif analysis['overall_sentiment'] == 'positive':
            recommendations.extend([
                'Match the positive energy',
                'Celebrate with the user',
                'Build on the enthusiasm',
                'Maintain the upbeat tone'
            ])
        else:
            recommendations.extend([
                'Maintain professional tone',
                'Provide clear information',
                'Be helpful and informative'
            ])
        
        return {
            'sentiment_analysis': analysis,
            'response_recommendations': recommendations,
            'suggested_tone': self._suggest_tone(analysis),
            'emotional_context': self._get_emotional_context(analysis)
        }
    
    def _suggest_tone(self, analysis: Dict) -> str:
        """Suggest appropriate response tone"""
        sentiment = analysis['overall_sentiment']
        intensity = analysis['intensity']
        
        if sentiment == 'negative':
            if intensity == 'strong':
                return 'very_supportive'
            else:
                return 'supportive'
        elif sentiment == 'positive':
            if intensity == 'strong':
                return 'enthusiastic'
            else:
                return 'friendly'
        else:
            return 'professional'
    
    def _get_emotional_context(self, analysis: Dict) -> Dict[str, Any]:
        """Get emotional context for better AI responses"""
        return {
            'user_mood': analysis['overall_sentiment'],
            'emotional_intensity': analysis['intensity'],
            'needs_support': analysis['overall_sentiment'] == 'negative' and analysis['confidence'] > 0.5,
            'celebration_opportunity': analysis['overall_sentiment'] == 'positive' and analysis['confidence'] > 0.7,
            'neutral_interaction': analysis['overall_sentiment'] == 'neutral'
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get sentiment analysis statistics"""
        if not self.sentiment_history:
            return {'message': 'No sentiment data available'}
        
        sentiments = [s['overall_sentiment'] for s in self.sentiment_history]
        
        return {
            'total_analyses': len(self.sentiment_history),
            'sentiment_distribution': {
                'positive': sentiments.count('positive'),
                'negative': sentiments.count('negative'),
                'neutral': sentiments.count('neutral')
            },
            'average_confidence': sum(s['confidence'] for s in self.sentiment_history) / len(self.sentiment_history),
            'recent_trend': self._calculate_trend(self.sentiment_history[-10:]) if len(self.sentiment_history) >= 10 else 'insufficient_data'
        }


# Global instance
sentiment_analyzer = EnhancedSentimentAnalyzer()

def analyze_text_sentiment(text: str) -> Dict[str, Any]:
    """Main function to analyze text sentiment"""
    return sentiment_analyzer.analyze_sentiment(text)

def get_sentiment_insights(text: str) -> Dict[str, Any]:
    """Get detailed sentiment insights for AI responses"""
    return sentiment_analyzer.get_sentiment_insights(text)

def analyze_conversation_mood(messages: List[str]) -> Dict[str, Any]:
    """Analyze overall conversation mood"""
    return sentiment_analyzer.analyze_conversation_mood(messages)

def get_sentiment_statistics() -> Dict[str, Any]:
    """Get sentiment analysis statistics"""
    return sentiment_analyzer.get_statistics()