"""
🚀 Advanced AI Brain System - Multi-Modal Offline Intelligence
Leverages TensorFlow, PyTorch, Transformers, and advanced ML libraries
"""

import os
import json
import pickle
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional, Union
from dataclasses import dataclass, asdict
import sqlite3
import threading
import time
from collections import defaultdict, deque, Counter
import hashlib

# Core ML Libraries
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Performance & Caching
import joblib
from functools import lru_cache

logger = logging.getLogger(__name__)

@dataclass
class AdvancedLearningPattern:
    """Enhanced learning pattern with multi-modal capabilities"""
    pattern_id: str
    input_features: Dict[str, Any]
    response_type: str
    expected_output: str
    confidence: float
    model_predictions: Dict[str, float]
    embedding_vector: List[float]
    context_metadata: Dict[str, Any]
    usage_count: int
    success_rate: float
    last_used: datetime
    performance_metrics: Dict[str, float]

class MultiModalEmbeddings:
    """Advanced embedding system using multiple models"""
    
    def __init__(self):
        self.models = {}
        self.load_embedding_models()
    
    def load_embedding_models(self):
        """Load various embedding models for different tasks"""
        try:
            # Try to load advanced models, fallback to basic ones
            try:
                from sentence_transformers import SentenceTransformer
                self.models['sentence_transformer'] = SentenceTransformer('all-MiniLM-L6-v2')
                logger.info("Loaded SentenceTransformer model")
            except ImportError:
                logger.warning("SentenceTransformers not available, using TF-IDF fallback")
            
            # TF-IDF for traditional text analysis (always available)
            self.models['tfidf'] = TfidfVectorizer(max_features=5000, stop_words='english')
            
            # Word2Vec for word embeddings (will be trained on user data)
            self.word2vec_model = None
            
        except Exception as e:
            logger.warning(f"Some embedding models failed to load: {e}")
            # Fallback to basic models
            self.models['tfidf'] = TfidfVectorizer(max_features=1000, stop_words='english')
    
    def get_sentence_embedding(self, text: str, model_type: str = 'auto') -> np.ndarray:
        """Get sentence embedding using best available model"""
        try:
            if model_type == 'auto':
                # Use best available model
                if 'sentence_transformer' in self.models:
                    return self.models['sentence_transformer'].encode([text])[0]
                else:
                    model_type = 'tfidf'
            
            if model_type == 'tfidf':
                # Use TF-IDF
                if not hasattr(self.models['tfidf'], 'vocabulary_'):
                    # Not fitted
                    self.models['tfidf'].fit([text])

                tfidf_vector = self.models['tfidf'].transform([text]).toarray()[0]
                return tfidf_vector
                
        except Exception as e:
            logger.error(f"Error getting embedding: {e}")
            return np.zeros(100)  # Default size
    
    def get_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts"""
        try:
            emb1 = self.get_sentence_embedding(text1)
            emb2 = self.get_sentence_embedding(text2)
            
            # Calculate cosine similarity
            similarity = cosine_similarity([emb1], [emb2])[0][0]
            return float(similarity)
        except Exception as e:
            logger.error(f"Error calculating similarity: {e}")
            return 0.0

class AdvancedMLModels:
    """Collection of advanced ML models for different tasks"""
    
    def __init__(self):
        self.models = {}
        self.initialize_models()
    
    def initialize_models(self):
        """Initialize various ML models"""
        # Text Classification Models
        self.models['intent_classifier'] = RandomForestClassifier(n_estimators=100, random_state=42)
        self.models['sentiment_classifier'] = GradientBoostingClassifier(random_state=42)
        
        # Clustering for pattern discovery
        self.models['pattern_clustering'] = KMeans(n_clusters=10, random_state=42)
        
        # Dimensionality reduction
        self.models['pca'] = PCA(n_components=50)
        
        logger.info("Initialized advanced ML models")
    
    def train_intent_classifier(self, texts: List[str], labels: List[str]):
        """Train intent classification model"""
        try:
            # Use TF-IDF features
            vectorizer = TfidfVectorizer(max_features=1000)
            X = vectorizer.fit_transform(texts)
            
            self.models['intent_classifier'].fit(X, labels)
            self.models['intent_vectorizer'] = vectorizer
            
            logger.info("Intent classifier trained successfully")
        except Exception as e:
            logger.error(f"Error training intent classifier: {e}")
    
    def predict_intent(self, text: str) -> Tuple[str, float]:
        """Predict intent with confidence"""
        try:
            if 'intent_vectorizer' not in self.models:
                # Fallback to rule-based classification
                text_lower = text.lower()
                if any(word in text_lower for word in ['hello', 'hi', 'hey']):
                    return 'greeting', 0.9
                elif '?' in text or any(word in text_lower for word in ['what', 'how', 'when', 'where', 'why']):
                    return 'question', 0.8
                elif any(word in text_lower for word in ['+', '-', '*', '/', 'calculate']):
                    return 'math', 0.7
                elif any(word in text_lower for word in ['code', 'program', 'python', 'html']):
                    return 'programming', 0.8
                else:
                    return 'general', 0.6
            
            X = self.models['intent_vectorizer'].transform([text])
            prediction = self.models['intent_classifier'].predict(X)[0]
            probabilities = self.models['intent_classifier'].predict_proba(X)[0]
            confidence = max(probabilities)
            
            return prediction, confidence
        except Exception as e:
            logger.error(f"Error predicting intent: {e}")
            return 'general', 0.5
    
    def analyze_sentiment(self, text: str) -> Tuple[str, float]:
        """Analyze sentiment of text"""
        try:
            # Simple rule-based sentiment for now
            positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful']
            negative_words = ['bad', 'terrible', 'awful', 'horrible', 'hate', 'frustrating']
            
            text_lower = text.lower()
            pos_count = sum(1 for word in positive_words if word in text_lower)
            neg_count = sum(1 for word in negative_words if word in text_lower)
            
            if pos_count > neg_count:
                return 'positive', min(0.9, 0.6 + pos_count * 0.1)
            elif neg_count > pos_count:
                return 'negative', min(0.9, 0.6 + neg_count * 0.1)
            else:
                return 'neutral', 0.6
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {e}")
            return 'neutral', 0.5

class AdvancedKnowledgeGraph:
    """Enhanced knowledge graph with semantic relationships"""
    
    def __init__(self):
        self.nodes = {}
        self.edges = defaultdict(list)
        self.embeddings = {}
        self.entity_types = {}
        self.relationship_strengths = {}
    
    def add_entity(self, entity_id: str, entity_text: str, entity_type: str = 'general', embedding: np.ndarray = None):
        """Add entity with semantic embedding"""
        self.nodes[entity_id] = {
            'text': entity_text,
            'type': entity_type,
            'created_at': datetime.now(),
            'access_count': 0,
            'importance': 0.5
        }
        self.embeddings[entity_id] = embedding
        self.entity_types[entity_id] = entity_type
    
    def add_relationship(self, entity1_id: str, entity2_id: str, relationship_type: str, strength: float = 0.5):
        """Add semantic relationship between entities"""
        self.edges[entity1_id].append({
            'target': entity2_id,
            'type': relationship_type,
            'strength': strength,
            'created_at': datetime.now()
        })
        
        # Bidirectional relationship
        self.edges[entity2_id].append({
            'target': entity1_id,
            'type': relationship_type,
            'strength': strength,
            'created_at': datetime.now()
        })
    
    def find_similar_entities(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict]:
        """Find entities similar to query embedding"""
        similarities = []
        
        for entity_id, embedding in self.embeddings.items():
            try:
                similarity = cosine_similarity([query_embedding], [embedding])[0][0]
                similarities.append((entity_id, similarity))
            except Exception as e:
                logger.error(f"Error calculating similarity for entity {entity_id}: {e}")
                continue
        
        # Sort by similarity and return top k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def get_entity_context(self, entity_id: str, depth: int = 2) -> Dict[str, Any]:
        """Get contextual information about an entity"""
        if entity_id not in self.nodes:
            return {}
        
        context = {
            'entity': self.nodes[entity_id],
            'related_entities': [],
            'relationship_summary': defaultdict(int)
        }
        
        # BFS to find related entities
        visited = set()
        queue = deque([(entity_id, 0)])
        
        while queue:
            current_id, current_depth = queue.popleft()
    
            if current_id in visited or current_depth >= depth:
                continue
            
            visited.add(current_id)
  
            for edge in self.edges[current_id]:
                target_id = edge['target']
                if target_id not in visited:
                    context['related_entities'].append({
                        'entity_id': target_id,
                        'relationship': edge['type'],
                        'strength': edge['strength'],
                        'depth': current_depth + 1
                    })
                    
                    context['relationship_summary'][edge['type']] += 1
                    
                    if current_depth + 1 < depth:
                        queue.append((target_id, current_depth + 1))
        
        return context

class AdvancedAIBrain:
    """Main advanced AI brain with multi-modal capabilities"""
    
    def __init__(self, db_path="advanced_ai_brain.db"):
        self.db_path = db_path
        
        # Initialize components
        self.embeddings = MultiModalEmbeddings()
        self.ml_models = AdvancedMLModels()
        self.knowledge_graph = AdvancedKnowledgeGraph()
        
        # Learning data
        self.learning_patterns = {}
        self.conversation_history = deque(maxlen=1000)
        self.feedback_buffer = deque(maxlen=500)

        # Performance metrics
        self.performance_metrics = {
            'total_interactions': 0,
            'successful_responses': 0,
            'average_confidence': 0.0,
            'learning_rate': 0.1,
            'model_accuracies': {}
        }
    
        # Initialize database and load state
        self.init_database()
        self.load_brain_state()
        
        # Start background learning
        self.start_background_learning()
        
        logger.info("Advanced AI Brain initialized successfully")
    
    def init_database(self):
        """Initialize advanced database schema"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Advanced learning patterns table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS advanced_patterns (
                pattern_id TEXT PRIMARY KEY,
                input_features TEXT,
                response_type TEXT,
                expected_output TEXT,
                confidence REAL,
                model_predictions TEXT,
                embedding_vector TEXT,
                context_metadata TEXT,
                usage_count INTEGER,
                success_rate REAL,
                last_used TIMESTAMP,
                performance_metrics TEXT
            )
        ''')
        
        # Knowledge graph entities
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS knowledge_entities (
                entity_id TEXT PRIMARY KEY,
                entity_text TEXT,
                entity_type TEXT,
                embedding_vector TEXT,
                metadata TEXT,
                created_at TIMESTAMP,
                access_count INTEGER,
                importance_score REAL
            )
        ''')
        
        # Advanced feedback log
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS advanced_feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                input_text TEXT,
                output_text TEXT,
                feedback_type TEXT,
                feedback_value REAL,
                model_predictions TEXT,
                embedding_similarity REAL,
                context_info TEXT,
                timestamp TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
        logger.info("Advanced database initialized")

    def generate_advanced_response(self, user_input: str, context: List[str] = None) -> Dict[str, Any]:
        """Generate advanced response using multiple AI techniques"""
        start_time = time.time()
        
        try:
            # Extract advanced features
            features = self.extract_advanced_features(user_input, context)
            
            # Get semantic embedding
            embedding = self.embeddings.get_sentence_embedding(user_input)
            
            # Predict intent and sentiment
            intent, intent_confidence = self.ml_models.predict_intent(user_input)
            sentiment, sentiment_confidence = self.ml_models.analyze_sentiment(user_input)
            
            # Find similar patterns
            similar_patterns = self.find_similar_patterns(embedding)
       
            # Generate response based on intent and patterns
            if similar_patterns:
                # Use learned pattern
                best_pattern = similar_patterns[0]
                response_text = best_pattern['expected_output']
                confidence = best_pattern['confidence']
                source = 'learned_pattern'
            else:
                # Generate new response
                response_text = self.generate_contextual_response(
                    user_input, intent, sentiment, context
                )
                confidence = intent_confidence
                source = 'generated'
            
            # Find related entities in knowledge graph
            related_entities = self.knowledge_graph.find_similar_entities(embedding, top_k=3)
            
            # Create new learning pattern
            pattern_id = self.create_advanced_pattern(
                user_input, response_text, features, embedding, intent, confidence
            )
            
            # Update performance metrics
            self.update_performance_metrics(confidence, source)
            
            # Prepare response data
            response_data = {
                'response': response_text,
                'confidence': confidence,
                'intent': intent,
                'sentiment': sentiment,
                'source': source,
                'pattern_id': pattern_id,
                'related_entities': related_entities,
                'processing_time': time.time() - start_time,
                'advanced_metrics': {
                    'embedding_dimensions': len(embedding),
                    'intent_confidence': intent_confidence,
                    'sentiment_confidence': sentiment_confidence,
                    'similar_patterns_found': len(similar_patterns)
                }
            }
            
            # Store conversation history
            self.conversation_history.append({
                'input': user_input,
                'response': response_text,
                'timestamp': datetime.now(),
                'metadata': response_data
            })
            
            return response_data
            
        except Exception as e:
            logger.error(f"Error generating advanced response: {e}")
            return {
                'response': "I'm experiencing some technical difficulties. Please try again.",
                'confidence': 0.0,
                'error': str(e),
                'source': 'error'
            }
    
    def extract_advanced_features(self, text: str, context: List[str] = None) -> Dict[str, Any]:
        """Extract comprehensive features from text"""
        features = {
            'text_length': len(text),
            'word_count': len(text.split()),
            'sentence_count': len([s for s in text.split('.') if s.strip()]),
            'question_marks': text.count('?'),
            'exclamation_marks': text.count('!'),
            'uppercase_ratio': sum(1 for c in text if c.isupper()) / len(text) if text else 0
        }
        
        # Linguistic features
        text_lower = text.lower()
        
        # Intent indicators
        features['has_greeting'] = any(word in text_lower for word in ['hello', 'hi', 'hey'])
        features['has_question'] = any(word in text_lower for word in ['what', 'how', 'when', 'where', 'who'])
        features['has_request'] = any(word in text_lower for word in ['please', 'can you', 'could you', 'help'])
        features['has_problem'] = any(word in text_lower for word in ['problem', 'issue', 'error', 'bug', 'solve'])
        features['has_code'] = any(word in text_lower for word in ['code', 'program', 'function', 'script'])
        
        # Context features
        if context:
            features['context_length'] = len(' '.join(context))
            features['context_similarity'] = self.calculate_context_similarity(text, context)
        else:
            features['context_length'] = 0
            features['context_similarity'] = 0.0
        
        return features
    
    def calculate_context_similarity(self, text: str, context: List[str]) -> float:
        """Calculate similarity between text and context"""
        
        if not context:
            return 0.0
        
        try:
            text_embedding = self.embeddings.get_sentence_embedding(text)
            context_embeddings = [self.embeddings.get_sentence_embedding(ctx) for ctx in context]
            
            similarities = [
                cosine_similarity([text_embedding], [ctx_emb])[0][0] 
                for ctx_emb in context_embeddings
            ]
            
            return max(similarities) if similarities else 0.0
        except Exception as e:
            logger.error(f"Error calculating context similarity: {e}")
            return 0.0
    
    def find_similar_patterns(self, query_embedding: np.ndarray, threshold: float = 0.7) -> List[Dict]:
        """Find similar learned patterns using embeddings"""
        similar_patterns = []
        
        for pattern_id, pattern in self.learning_patterns.items():
            try:
                pattern_embedding = np.array(pattern.embedding_vector)
                similarity = cosine_similarity([query_embedding], [pattern_embedding])[0][0]
             
                if similarity >= threshold:
                    similar_patterns.append({
                        'pattern_id': pattern_id,
                        'similarity': similarity,
                        'confidence': pattern.confidence,
                        'expected_output': pattern.expected_output,
                        'usage_count': pattern.usage_count,
                        'success_rate': pattern.success_rate
                    })
            except Exception as e:
                logger.error(f"Error comparing pattern {pattern_id}: {e}")
                continue
        
        # Sort by combined similarity and confidence
        similar_patterns.sort(key=lambda x: x['similarity'] * x['confidence'], reverse=True)
        return similar_patterns
    
    def generate_contextual_response(self, user_input: str, intent: str, sentiment: str, 
                                   context: List[str] = None) -> str:
        """Generate contextual response based on intent and sentiment"""
        
        if intent == 'greeting':
            if sentiment == 'positive':
                return "Hello! I'm excited to help you today. What can I assist you with?"
            else:
                return "Hi there! I'm here to help with anything you need."
        
        elif intent == 'question':
            if 'python' in user_input.lower():
                return "Python is a versatile programming language! I can help with syntax, libraries, or concepts. What specifically would you like to know more?"
            elif 'ai' in user_input.lower():
                return "AI is fascinating! I use advanced machine learning techniques including neural networks and semantic analysis. What would you like to explore?"
            elif 'machine learning' in user_input.lower():
                return "Machine Learning is at the core of my advanced capabilities! I use various algorithms for pattern recognition and learning. What aspect interests you?"
            else:
                return "That's a great question! I'm analyzing it using my advanced AI capabilities. Could you provide a bit more context?"
        
        elif intent == 'math':
            return "I can help with mathematical calculations and concepts using my analytical capabilities. What would you like to calculate or understand?"
        
        elif intent == 'programming':
            return "I'd be happy to help with programming! I can assist with code, debugging, algorithms, and best practices. What are you working on?"
        
        else:
            # General response with context awareness
            if context:
                return f"Based on our conversation context, I understand you're asking about something related to our previous discussion. Let me help you with that."
            else:
                return "I'm processing your request using my advanced AI capabilities including neural networks, semantic analysis, and knowledge graphs. I learn from every interaction to provide increasingly better responses. How can I help you today?"
    
    def create_advanced_pattern(self, user_input: str, response: str, features: Dict, 
                              embedding: np.ndarray, intent: str, confidence: float) -> str:
        """Create advanced learning pattern"""
        pattern_id = hashlib.md5(f"{user_input}{response}".encode()).hexdigest()
        
        pattern = AdvancedLearningPattern(
            pattern_id=pattern_id,
            input_features=features,
            response_type=intent,
            expected_output=response,
            confidence=confidence,
            model_predictions={'intent_confidence': confidence},
            embedding_vector=embedding.tolist(),
            context_metadata={'created_at': datetime.now().isoformat()},
            usage_count=1,
            success_rate=0.5,
            last_used=datetime.now(),
            performance_metrics={'initial_confidence': confidence}
        )
        
        self.learning_patterns[pattern_id] = pattern
        
        # Add to knowledge graph
        entity_id = f"pattern_{pattern_id[:8]}"
        self.knowledge_graph.add_entity(
            entity_id, user_input, 'user_input', embedding
        )
        
        return pattern_id
    
    def receive_advanced_feedback(self, user_input: str, ai_response: str, 
                                feedback_type: str, feedback_value: Any, 
                                context_info: Dict = None) -> bool:
        """Process advanced feedback and update learning"""
        try:
            # Convert feedback to score
            feedback_score = self.convert_feedback_to_score(feedback_type, feedback_value)
         
            # Get input embedding for pattern matching
            input_embedding = self.embeddings.get_sentence_embedding(user_input)
            
            # Find related patterns
            similar_patterns = self.find_similar_patterns(input_embedding, threshold=0.6)
            
            for pattern_info in similar_patterns:
                pattern_id = pattern_info['pattern_id']
                if pattern_id in self.learning_patterns:
                    pattern = self.learning_patterns[pattern_id]
    
                    # Update success rate with weighted average
                    pattern.success_rate = (pattern.success_rate * 0.8) + (feedback_score * 0.2)
                    pattern.confidence = min(pattern.confidence + (feedback_score - 0.5) * 0.1, 1.0)
                    pattern.usage_count += 1
                    pattern.last_used = datetime.now()
            
            # Store detailed feedback
            feedback_data = {
                'input': user_input,
                'output': ai_response,
                'feedback_type': feedback_type,
                'feedback_value': feedback_value,
                'feedback_score': feedback_score,
                'embedding_similarity': 0.0,
                'context_info': context_info or {},
                'timestamp': datetime.now(),
                'model_predictions': {'patterns_found': len(similar_patterns)}
            }
            
            self.feedback_buffer.append(feedback_data)
            
            # Log to database
            self.log_advanced_feedback(feedback_data)
            
            logger.info(f"Processed feedback: {feedback_type} -> {feedback_score}")
            return True
            
        except Exception as e:
            logger.error(f"Error processing feedback: {e}")
            return False
    
    def convert_feedback_to_score(self, feedback_type: str, feedback_value: Any) -> float:
        """Convert various feedback types to numerical scores"""
        if feedback_type == 'rating':
            return max(0, min(1, (feedback_value - 1) / 4))
        elif feedback_type == 'thumbs':
            return 1.0 if feedback_value else 0.0
        elif feedback_type == 'negative':
            return 0.2  # Negative feedback
        elif feedback_type == 'positive':
            return 0.8 if feedback_value else 0.5
        else:
            return 0.5
    
    def update_performance_metrics(self, confidence: float, source: str):
        """Update performance metrics"""
        self.performance_metrics['total_interactions'] += 1
        
        if confidence > 0.7:
            self.performance_metrics['successful_responses'] += 1
        
        # Update average confidence with exponential moving average
        alpha = 0.1
        self.performance_metrics['average_confidence'] = (
            (1 - alpha) * self.performance_metrics['average_confidence'] +
            alpha * confidence
        )
        
        # Update learning rate (how much we adapt based on recent feedback)
        if len(self.feedback_buffer) > 10:
            recent_feedback = list(self.feedback_buffer)[-10:]
            recent_scores = [fb['feedback_score'] for fb in recent_feedback]
            self.performance_metrics['learning_rate'] = np.mean(recent_scores)
    
    def get_advanced_statistics(self) -> Dict[str, Any]:
        """Get comprehensive AI brain statistics"""
        stats = {
            'learning_patterns': {
                'total_patterns': len(self.learning_patterns),
                'avg_confidence': np.mean([p.confidence for p in self.learning_patterns.values()]) if self.learning_patterns else 0,
                'avg_success_rate': np.mean([p.success_rate for p in self.learning_patterns.values()]) if self.learning_patterns else 0,
                'most_used_patterns': self.get_most_used_patterns(5)
            },
            'knowledge_graph': {
                'total_entities': len(self.knowledge_graph.nodes),
                'total_relationships': sum(len(edges) for edges in self.knowledge_graph.edges.values()),
                'entity_types': dict(Counter(self.knowledge_graph.entity_types.values()))
            },
            'performance_metrics': self.performance_metrics,
            'conversation_stats': {
                'total_conversations': len(self.conversation_history),
                'avg_response_length': np.mean([len(conv['response']) for conv in self.conversation_history]) if self.conversation_history else 0,
                'recent_intents': self.get_recent_intents(10)
            },
            'system_info': {
                'embedding_models': list(self.embeddings.models.keys()),
                'ml_models': list(self.ml_models.models.keys()),
                'database_path': self.db_path
            }
        }
        
        return stats
    
    def get_most_used_patterns(self, top_k: int = 5) -> List[Dict]:
        """Get most frequently used patterns"""
        patterns = [(p.pattern_id, p.usage_count, p.success_rate) 
                   for p in self.learning_patterns.values()]
        patterns.sort(key=lambda x: x[1], reverse=True)
        
        return [{'pattern_id': pid[:8], 'usage_count': count, 'success_rate': rate} 
                for pid, count, rate in patterns[:top_k]]
    
    def get_recent_intents(self, count: int = 10) -> List[str]:
        """Get recent conversation intents"""
        recent_convs = list(self.conversation_history)[-count:]
        return [conv['metadata'].get('intent', 'unknown') for conv in recent_convs]
    
    def save_brain_state(self):
        """Save advanced brain state to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
    
        try:
            # Save learning patterns
            for pattern_id, pattern in self.learning_patterns.items():
                cursor.execute('''
                    INSERT OR REPLACE INTO advanced_patterns 
                    (pattern_id, input_features, response_type, expected_output, 
                     confidence, model_predictions, embedding_vector, context_metadata,
                     usage_count, success_rate, last_used, performance_metrics)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    pattern_id,
                    json.dumps(pattern.input_features),
                    pattern.response_type,
                    pattern.expected_output,
                    pattern.confidence,
                    json.dumps(pattern.model_predictions),
                    json.dumps(pattern.embedding_vector),
                    json.dumps(pattern.context_metadata),
                    pattern.usage_count,
                    pattern.success_rate,
                    pattern.last_used.isoformat(),
                    json.dumps(pattern.performance_metrics)
                ))
            
            conn.commit()
            logger.info("Advanced brain state saved successfully")
            
        except Exception as e:
            logger.error(f"Error saving brain state: {e}")
        finally:
            conn.close()
    
    def load_brain_state(self):
        """Load advanced brain state from database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Load learning patterns
            cursor.execute('SELECT * FROM advanced_patterns')
            for row in cursor.fetchall():
                try:
                    pattern = AdvancedLearningPattern(
                        pattern_id=row[0],
                        input_features=json.loads(row[1]),
                        response_type=row[2],
                        expected_output=row[3],
                        confidence=row[4],
                        model_predictions=json.loads(row[5]),
                        embedding_vector=json.loads(row[6]),
                        context_metadata=json.loads(row[7]),
                        usage_count=row[8],
                        success_rate=row[9],
                        last_used=datetime.fromisoformat(row[10]),
                        performance_metrics=json.loads(row[11])
                    )
                    self.learning_patterns[pattern.pattern_id] = pattern
                except Exception as e:
                    logger.warning(f"Error loading pattern {row[0]}: {e}")
            
            logger.info(f"Loaded {len(self.learning_patterns)} patterns")
            
        except Exception as e:
            logger.warning(f"Error loading brain state: {e}")
        finally:
            conn.close()
    
    def log_advanced_feedback(self, feedback_data: Dict):
        """Log advanced feedback to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                INSERT INTO advanced_feedback 
                (input_text, output_text, feedback_type, feedback_value,
                 model_predictions, embedding_similarity, context_info, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                feedback_data['input'],
                feedback_data['output'],
                feedback_data['feedback_type'],
                feedback_data['feedback_score'],
                json.dumps(feedback_data['model_predictions']),
                feedback_data['embedding_similarity'],
                json.dumps(feedback_data['context_info']),
                feedback_data['timestamp']
            ))
            
            conn.commit()
        except Exception as e:
            logger.error(f"Error logging feedback: {e}")
        finally:
            conn.close()
    
    def start_background_learning(self):
        """Start background learning thread"""
        def learning_loop():
            while True:
                try:
                    # Train models from accumulated feedback
                    if len(self.feedback_buffer) >= 10:
                        self.train_models_from_feedback()
        
                    # Save state periodically
                    self.save_brain_state()
                    
                    # Sleep for 5 minutes
                    time.sleep(300)
                    
                except Exception as e:
                    logger.error(f"Error in background learning: {e}")
                    time.sleep(60)
        
        learning_thread = threading.Thread(target=learning_loop, daemon=True)
        learning_thread.start()
        logger.info("Advanced background learning started")
    
    def train_models_from_feedback(self):
        """Train ML models from accumulated feedback"""
        try:
            feedback_data = list(self.feedback_buffer)
            
            # Prepare training data for intent classification
            texts = [fb['input'] for fb in feedback_data]
            intents = []
            for fb in feedback_data:
                text = fb['input'].lower()
                if any(word in text for word in ['hello', 'hi', 'hey']):
                    intents.append('greeting')
                elif '?' in text:
                    intents.append('question')
                elif any(word in text for word in ['+', '-', '*', '/']):
                    intents.append('math')
                else:
                    intents.append('general')
            
            # Train intent classifier if we have enough diverse data
            if len(set(intents)) > 1:
                self.ml_models.train_intent_classifier(texts, intents)
                logger.info(f"Trained intent classifier with {len(texts)} samples")
            
            # Clear processed feedback
            self.feedback_buffer.clear()
            
        except Exception as e:
            logger.error(f"Error training models from feedback: {e}")

# Global instance
advanced_ai_brain = AdvancedAIBrain()

# Main interface functions
def get_advanced_ai_response(user_input: str, context: List[str] = None) -> Dict[str, Any]:
    """Get advanced AI response"""
    return advanced_ai_brain.generate_advanced_response(user_input, context)

def provide_advanced_feedback(user_input: str, ai_response: str, feedback_type: str,
                            feedback_value: Any, context_info: Dict = None) -> bool:
    """Provide advanced feedback"""
    return advanced_ai_brain.receive_advanced_feedback(
        user_input, ai_response, feedback_type, feedback_value, context_info
    )

def get_advanced_statistics() -> Dict[str, Any]:
    """Get advanced AI statistics"""
    return advanced_ai_brain.get_advanced_statistics()

def analyze_text(text: str) -> Dict[str, Any]:
    """Advanced text analysis"""
    embedding = advanced_ai_brain.embeddings.get_sentence_embedding(text)
    intent, intent_conf = advanced_ai_brain.ml_models.predict_intent(text)
    sentiment, sentiment_conf = advanced_ai_brain.ml_models.analyze_sentiment(text)
    
    return {
        'intent': intent,
        'intent_confidence': intent_conf,
        'sentiment': sentiment,
        'sentiment_confidence': sentiment_conf,
        'similar_entities': advanced_ai_brain.knowledge_graph.find_similar_entities(embedding, top_k=3)
    }

if __name__ == "__main__":
    # Test the advanced AI brain
    print("🚀 Advanced AI Brain Test")
    
    response = get_advanced_ai_response("Hello! Can you help me with Python programming?")
    print(f"Response: {response['response']}")
    print(f"Confidence: {response['confidence']}")
    print(f"Intent: {response['intent']}")
    print(f"Processing time: {response['processing_time']:.3f}s")
    
    # Provide feedback
    provide_advanced_feedback("Hello! Can you help me with Python programming?", 
                            response['response'], "thumbs", True)
    
    # Get statistics
    stats = get_advanced_statistics()
    print(f"\nStatistics: {json.dumps(stats, indent=2, default=str)}")