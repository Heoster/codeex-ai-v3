"""
🧠 Enhanced AI Brain System - Self-Training & Adaptive Learning
Advanced AI that learns from interactions and improves over time
"""

import json
import sqlite3
import numpy as np
import pickle
import os
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, asdict
import logging
from collections import defaultdict, deque
import threading
import time
import hashlib

logger = logging.getLogger(__name__)


@dataclass
class LearningPattern:
    """Represents a learned pattern from user interactions"""
    pattern_id: str
    input_features: List[float]
    expected_output: str
    confidence: float
    usage_count: int
    last_used: datetime
    success_rate: float
    context_tags: List[str]


@dataclass
class MemoryNode:
    """Represents a memory node in the knowledge graph"""
    node_id: str
    content: str
    connections: List[str]
    strength: float
    created_at: datetime
    access_count: int
    importance_score: float


class AdaptiveNeuralNetwork:
    """Self-adapting neural network that learns from feedback"""

    def __init__(self, input_size=100, hidden_size=64, output_size=32):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = 0.001
        self.momentum = 0.9

        # Initialize weights with Xavier initialization
        self.weights = {
            'W1': np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size),
            'b1': np.zeros((1, hidden_size)),
            'W2': np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / hidden_size),
            'b2': np.zeros((1, output_size))
        }

        # Momentum terms
        self.velocity = {key: np.zeros_like(val)
                         for key, val in self.weights.items()}

        # Performance tracking
        self.training_history = []
        self.accuracy_history = []

    def relu(self, x):
        """ReLU activation function"""
        return np.maximum(0, x)

    def relu_derivative(self, x):
        """Derivative of ReLU"""
        return (x > 0).astype(float)

    def softmax(self, x):
        """Softmax activation for output layer"""
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def forward(self, X):
        """Forward pass through the network"""
        self.z1 = np.dot(X, self.weights['W1']) + self.weights['b1']
        self.a1 = self.relu(self.z1)
        self.z2 = np.dot(self.a1, self.weights['W2']) + self.weights['b2']
        self.a2 = self.softmax(self.z2)
        return self.a2

    def backward(self, X, y, output):
        """Backward pass with gradient descent"""
        m = X.shape[0]

        # Output layer gradients
        dz2 = output - y
        dW2 = (1/m) * np.dot(self.a1.T, dz2)
        db2 = (1/m) * np.sum(dz2, axis=0, keepdims=True)

        # Hidden layer gradients
        da1 = np.dot(dz2, self.weights['W2'].T)
        dz1 = da1 * self.relu_derivative(self.z1)
        dW1 = (1/m) * np.dot(X.T, dz1)
        db1 = (1/m) * np.sum(dz1, axis=0, keepdims=True)

        # Update weights with momentum
        gradients = {'W1': dW1, 'b1': db1, 'W2': dW2, 'b2': db2}

        for key in self.weights:
            self.velocity[key] = self.momentum * \
                self.velocity[key] - self.learning_rate * gradients[key]
            self.weights[key] += self.velocity[key]

    def train(self, X, y, epochs=100):
        """Train the network on data"""
        for epoch in range(epochs):
            output = self.forward(X)
            loss = self.compute_loss(y, output)
            self.backward(X, y, output)

            if epoch % 10 == 0:
                accuracy = self.compute_accuracy(y, output)
                self.training_history.append(loss)
                self.accuracy_history.append(accuracy)
                logger.info(
                    f"Epoch {epoch}: Loss = {loss:.4f}, Accuracy = {accuracy:.4f}")

    def compute_loss(self, y_true, y_pred):
        """Compute cross-entropy loss"""
        return -np.mean(np.sum(y_true * np.log(y_pred + 1e-8), axis=1))

    def compute_accuracy(self, y_true, y_pred):
        """Compute prediction accuracy"""
        predictions = np.argmax(y_pred, axis=1)
        true_labels = np.argmax(y_true, axis=1)
        return np.mean(predictions == true_labels)


class KnowledgeGraph:
    """Dynamic knowledge graph that builds connections between concepts"""

    def __init__(self):
        self.nodes = {}
        self.connections = defaultdict(list)
        self.concept_embeddings = {}

    def add_node(self, node: MemoryNode):
        """Add a new memory node to the graph"""
        self.nodes[node.node_id] = node

    def create_connection(self, node1_id: str, node2_id: str, strength: float):
        """Create a connection between two nodes"""
        self.connections[node1_id].append((node2_id, strength))
        self.connections[node2_id].append((node1_id, strength))

    def find_related_concepts(self, node_id: str, max_depth=3) -> List[str]:
        """Find related concepts using graph traversal"""
        visited = set()
        queue = deque([(node_id, 0)])
        related = []

        while queue:
            current_id, depth = queue.popleft()
            if current_id in visited or depth > max_depth:
                continue

            visited.add(current_id)
            if depth > 0:  # Don't include the starting node
                related.append(current_id)

            # Add connected nodes to queue
            for connected_id, strength in self.connections.get(current_id, []):
                if connected_id not in visited and strength > 0.3:  # Threshold for relevance
                    queue.append((connected_id, depth + 1))

        return related

    def update_node_importance(self, node_id: str, feedback_score: float):
        """Update node importance based on feedback"""
        if node_id in self.nodes:
            node = self.nodes[node_id]
            node.access_count += 1
            node.importance_score = (
                node.importance_score * 0.9) + (feedback_score * 0.1)


class SelfTrainingAI:
    """Main AI brain that learns and adapts from interactions"""

    def __init__(self, db_path="ai_brain.db"):
        self.db_path = db_path
        self.neural_network = AdaptiveNeuralNetwork()
        self.knowledge_graph = KnowledgeGraph()
        self.learning_patterns = {}
        self.feedback_buffer = deque(maxlen=1000)
        self.training_thread = None
        self.is_training = False

        # Initialize database
        self.init_database()
        self.load_brain_state()

        # Start background learning
        self.start_background_learning()

    def init_database(self):
        """Initialize SQLite database for persistent storage"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Learning patterns table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS learning_patterns (
                pattern_id TEXT PRIMARY KEY,
                input_features TEXT,
                expected_output TEXT,
                confidence REAL,
                usage_count INTEGER,
                last_used TIMESTAMP,
                success_rate REAL,
                context_tags TEXT
            )
        ''')

        # Memory nodes table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS memory_nodes (
                node_id TEXT PRIMARY KEY,
                content TEXT,
                connections TEXT,
                strength REAL,
                created_at TIMESTAMP,
                access_count INTEGER,
                importance_score REAL
            )
        ''')

        # Feedback table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS feedback_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                input_text TEXT,
                output_text TEXT,
                feedback_score REAL,
                timestamp TIMESTAMP,
                context_info TEXT
            )
        ''')

        # Neural network weights table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS network_weights (
                layer_name TEXT PRIMARY KEY,
                weights_data BLOB,
                updated_at TIMESTAMP
            )
        ''')

        conn.commit()
        conn.close()

    def save_brain_state(self):
        """Save current brain state to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Save neural network weights
        for layer_name, weights in self.neural_network.weights.items():
            weights_blob = pickle.dumps(weights)
            cursor.execute('''
                INSERT OR REPLACE INTO network_weights (layer_name, weights_data, updated_at)
                VALUES (?, ?, ?)
            ''', (layer_name, weights_blob, datetime.now()))

        # Save learning patterns
        for pattern_id, pattern in self.learning_patterns.items():
            cursor.execute('''
                INSERT OR REPLACE INTO learning_patterns 
                (pattern_id, input_features, expected_output, confidence, usage_count, 
                 last_used, success_rate, context_tags)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                pattern_id,
                json.dumps(pattern.input_features),
                pattern.expected_output,
                pattern.confidence,
                pattern.usage_count,
                pattern.last_used,
                pattern.success_rate,
                json.dumps(pattern.context_tags)
            ))

        # Save memory nodes
        for node_id, node in self.knowledge_graph.nodes.items():
            cursor.execute('''
                INSERT OR REPLACE INTO memory_nodes
                (node_id, content, connections, strength, created_at, access_count, importance_score)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                node_id,
                node.content,
                json.dumps(node.connections),
                node.strength,
                node.created_at,
                node.access_count,
                node.importance_score
            ))

        conn.commit()
        conn.close()
        logger.info("Brain state saved successfully")

    def load_brain_state(self):
        """Load brain state from database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            # Load neural network weights
            cursor.execute(
                'SELECT layer_name, weights_data FROM network_weights')
            for layer_name, weights_blob in cursor.fetchall():
                self.neural_network.weights[layer_name] = pickle.loads(
                    weights_blob)

            # Load learning patterns
            cursor.execute('SELECT * FROM learning_patterns')
            for row in cursor.fetchall():
                pattern = LearningPattern(
                    pattern_id=row[0],
                    input_features=json.loads(row[1]),
                    expected_output=row[2],
                    confidence=row[3],
                    usage_count=row[4],
                    last_used=datetime.fromisoformat(
                        row[5]) if row[5] else datetime.now(),
                    success_rate=row[6],
                    context_tags=json.loads(row[7])
                )
                self.learning_patterns[pattern.pattern_id] = pattern

            # Load memory nodes
            cursor.execute('SELECT * FROM memory_nodes')
            for row in cursor.fetchall():
                node = MemoryNode(
                    node_id=row[0],
                    content=row[1],
                    connections=json.loads(row[2]),
                    strength=row[3],
                    created_at=datetime.fromisoformat(
                        row[4]) if row[4] else datetime.now(),
                    access_count=row[5],
                    importance_score=row[6]
                )
                self.knowledge_graph.add_node(node)

            logger.info("Brain state loaded successfully")

        except Exception as e:
            logger.warning(f"Could not load brain state: {e}")

        conn.close()

    def extract_features(self, text: str) -> List[float]:
        """Extract numerical features from text for neural network"""
        features = []
        text_lower = text.lower()

        # Basic text statistics
        features.extend([
            len(text) / 1000.0,  # Normalized text length
            len(text.split()) / 100.0,  # Normalized word count
            text.count('?') / 10.0,  # Question indicators
            text.count('!') / 10.0,  # Exclamation indicators
            text.count('.') / 20.0,  # Sentence indicators
        ])

        # Semantic categories
        categories = {
            'math': ['calculate', 'solve', 'math', 'number', '+', '-', '*', '/', 'sqrt', 'sin', 'cos'],
            'programming': ['code', 'program', 'function', 'python', 'javascript', 'html', 'css'],
            'knowledge': ['what', 'who', 'where', 'when', 'how', 'why', 'tell', 'explain'],
            'greeting': ['hello', 'hi', 'hey', 'greetings', 'good morning', 'good evening'],
            'creation': ['create', 'make', 'build', 'generate', 'design', 'develop']
        }

        for category, keywords in categories.items():
            score = sum(
                1 for keyword in keywords if keyword in text_lower) / len(keywords)
            features.append(score)

        # Emotional tone indicators
        positive_words = ['good', 'great', 'excellent',
                          'amazing', 'wonderful', 'fantastic']
        negative_words = ['bad', 'terrible',
                          'awful', 'horrible', 'disappointing']

        features.extend([
            sum(1 for word in positive_words if word in text_lower) /
            len(positive_words),
            sum(1 for word in negative_words if word in text_lower) /
            len(negative_words)
        ])

        # Pad or truncate to fixed size
        while len(features) < 100:
            features.append(0.0)

        return features[:100]

    def generate_response(self, user_input: str, context: List[str] = None) -> str:
        """Generate intelligent response using learned patterns"""
        # Extract features
        input_features = self.extract_features(user_input)

        # Check for learned patterns first
        pattern_response = self.check_learned_patterns(
            input_features, user_input)
        if pattern_response:
            return pattern_response

        # Use neural network for classification
        features_array = np.array([input_features])
        network_output = self.neural_network.forward(features_array)

        # Generate contextual response
        response = self.generate_contextual_response(
            user_input, network_output[0], context)

        # Create new learning pattern
        self.create_learning_pattern(input_features, response, user_input)

        return response

    def check_learned_patterns(self, input_features: List[float], user_input: str) -> Optional[str]:
        """Check if we have a learned pattern for similar input"""
        best_match = None
        best_similarity = 0.0

        for pattern in self.learning_patterns.values():
            similarity = self.calculate_similarity(
                input_features, pattern.input_features)
            if similarity > best_similarity and similarity > 0.8:  # High similarity threshold
                best_similarity = similarity
                best_match = pattern

        if best_match and best_match.confidence > 0.7:
            # Update usage statistics
            best_match.usage_count += 1
            best_match.last_used = datetime.now()
            return best_match.expected_output

        return None

    def calculate_similarity(self, features1: List[float], features2: List[float]) -> float:
        """Calculate cosine similarity between feature vectors"""
        if len(features1) != len(features2):
            return 0.0

        dot_product = sum(a * b for a, b in zip(features1, features2))
        magnitude1 = sum(a * a for a in features1) ** 0.5
        magnitude2 = sum(b * b for b in features2) ** 0.5

        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0

        return dot_product / (magnitude1 * magnitude2)

    def generate_contextual_response(self, user_input: str, network_output: np.ndarray, context: List[str] = None) -> str:
        """Generate response based on neural network output and context"""
        # Determine response category based on network output
        category_index = np.argmax(network_output)
        confidence = float(network_output[category_index])

        categories = ['math', 'programming', 'knowledge',
                      'greeting', 'creation', 'general']
        category = categories[min(category_index, len(categories) - 1)]

        # Generate response based on category
        if category == 'math':
            return self.handle_math_query(user_input)
        elif category == 'programming':
            return self.handle_programming_query(user_input)
        elif category == 'knowledge':
            return self.handle_knowledge_query(user_input)
        elif category == 'greeting':
            return self.handle_greeting(user_input)
        elif category == 'creation':
            return self.handle_creation_request(user_input)
        else:
            return self.handle_general_query(user_input, context)

    def handle_math_query(self, query: str) -> str:
        """Handle mathematical queries with learning"""
        # Basic math operations
        import re

        # Addition
        if '+' in query:
            numbers = re.findall(r'\d+(?:\.\d+)?', query)
            if len(numbers) >= 2:
                result = sum(float(n) for n in numbers)
                return f"The sum is {result}. I've learned this calculation pattern!"

        # Multiplication
        elif '*' in query or '×' in query:
            numbers = re.findall(r'\d+(?:\.\d+)?', query)
            if len(numbers) >= 2:
                result = float(numbers[0]) * float(numbers[1])
                return f"The product is {result}. Mathematical pattern stored for future reference!"

        return "I can help with mathematical calculations. I'm learning your preferred calculation styles!"

    def handle_programming_query(self, query: str) -> str:
        """Handle programming queries with adaptive learning"""
        query_lower = query.lower()

        if 'python' in query_lower:
            return "Python is excellent for AI development! I'm continuously learning Python patterns from our conversations."
        elif 'javascript' in query_lower:
            return "JavaScript powers modern web development. I adapt my responses based on your JavaScript interests!"
        elif 'create' in query_lower and 'website' in query_lower:
            return "I can help create websites! My responses improve as I learn your coding preferences."

        return "I'm here to help with programming! I learn from each coding question to provide better assistance."

    def handle_knowledge_query(self, query: str) -> str:
        """Handle knowledge queries with memory building"""
        # Build knowledge connections
        self.build_knowledge_connections(query)

        query_lower = query.lower()

        if 'capital' in query_lower:
            return "I can help with geography questions! I'm building a knowledge graph of world facts."
        elif 'president' in query_lower or 'prime minister' in query_lower:
            return "I can provide information about world leaders! My knowledge base expands with each query."

        return "I'm learning to provide better knowledge-based answers! Each question helps me improve."

    def handle_greeting(self, greeting: str) -> str:
        """Handle greetings with personality learning"""
        greetings = [
            "Hello! I'm CODEEX AI, developed by Heoster. I'm learning to be more helpful with each conversation!",
            "Hi there! I'm an advanced AI assistant created by the talented developers at Heoster. My responses get better as we interact more!",
            "Greetings! I'm CODEEX AI by Heoster - an AI that learns and adapts from our chats. How can I assist you today?",
            "Welcome! I'm powered by Heoster's cutting-edge AI technology. I'm here to help and learn from our interactions!"
        ]
        return greetings[hash(greeting) % len(greetings)]

    def handle_creation_request(self, request: str) -> str:
        """Handle creation requests with pattern learning"""
        return "I can help create various things! I learn your preferences to provide more personalized solutions."

    def handle_general_query(self, query: str, context: List[str] = None) -> str:
        """Handle general queries with context awareness"""
        query_lower = query.lower()

        # Developer/creator queries
        if any(word in query_lower for word in ['developer', 'creator', 'made', 'built', 'author', 'who created', 'who made']):
            return """I was created by a talented developer who specializes in AI and machine learning technologies. This CODEEX AI system represents their expertise in:

🚀 **Advanced AI Development**: Neural networks, deep learning, and conversational AI
🧠 **Machine Learning**: Reinforcement learning, natural language processing, and intelligent systems  
🔒 **Security**: Encrypted storage, secure authentication, and privacy-focused design
💻 **Full-Stack Development**: Python, Flask, modern web technologies, and database systems
🎯 **SEO & Marketing**: High-performance web applications with enterprise-level optimization

The developer has crafted this system to be both powerful and user-friendly, combining cutting-edge AI research with practical applications. Every feature, from the context memory to the encrypted chat history, reflects their commitment to creating intelligent, secure, and innovative software solutions."""

        # About the app queries
        elif any(word in query_lower for word in ['about', 'what is this', 'this app', 'codeex', 'heoster']):
            return """CODEEX AI by Heoster is an advanced conversational AI platform that represents the future of intelligent assistance. Here's what makes it special:

🧠 **Intelligent Context Memory**: Remembers our conversations and learns from interactions
🔐 **Military-Grade Security**: AES-256 encryption protects all your data
🚀 **Advanced AI Brain**: Uses multiple AI models including neural networks and machine learning
🏋️ **Self-Improving**: Reinforcement learning helps me get better over time
📚 **Knowledge Integration**: Connected to Wikipedia and other knowledge sources
💬 **Natural Conversations**: Designed to feel like talking to a knowledgeable friend

Built by a skilled developer who combines AI research with practical software engineering, this system showcases the potential of modern artificial intelligence while maintaining the highest standards of security and user privacy."""

        return "I'm learning to understand you better! Each interaction helps me provide more relevant responses. Feel free to ask me about my creator, this app, or anything else you'd like to know!"

    def build_knowledge_connections(self, query: str):
        """Build connections in the knowledge graph"""
        # Extract key concepts from query
        concepts = self.extract_concepts(query)

        for concept in concepts:
            node_id = hashlib.md5(concept.encode()).hexdigest()

            if node_id not in self.knowledge_graph.nodes:
                node = MemoryNode(
                    node_id=node_id,
                    content=concept,
                    connections=[],
                    strength=1.0,
                    created_at=datetime.now(),
                    access_count=1,
                    importance_score=0.5
                )
                self.knowledge_graph.add_node(node)
            else:
                self.knowledge_graph.nodes[node_id].access_count += 1

    def extract_concepts(self, text: str) -> List[str]:
        """Extract key concepts from text"""
        # Simple concept extraction (can be enhanced with NLP)
        words = text.lower().split()
        concepts = []

        # Filter out common words and extract meaningful concepts
        stop_words = {'the', 'is', 'at', 'which', 'on',
                      'a', 'an', 'and', 'or', 'but', 'in', 'with'}

        for word in words:
            if len(word) > 3 and word not in stop_words:
                concepts.append(word)

        return concepts

    def create_learning_pattern(self, input_features: List[float], response: str, original_input: str):
        """Create a new learning pattern"""
        pattern_id = hashlib.md5(
            (original_input + response).encode()).hexdigest()

        pattern = LearningPattern(
            pattern_id=pattern_id,
            input_features=input_features,
            expected_output=response,
            confidence=0.5,  # Initial confidence
            usage_count=1,
            last_used=datetime.now(),
            success_rate=0.5,  # Initial success rate
            context_tags=self.extract_concepts(original_input)
        )

        self.learning_patterns[pattern_id] = pattern

    def receive_feedback(self, user_input: str, ai_response: str, feedback_score: float):
        """Receive feedback and learn from it"""
        # Store feedback
        feedback_data = {
            'input': user_input,
            'output': ai_response,
            'score': feedback_score,
            'timestamp': datetime.now(),
            'features': self.extract_features(user_input)
        }

        self.feedback_buffer.append(feedback_data)

        # Update learning patterns
        self.update_patterns_from_feedback(feedback_data)

        # Log feedback to database
        self.log_feedback(user_input, ai_response, feedback_score)

    def update_patterns_from_feedback(self, feedback_data: Dict):
        """Update learning patterns based on feedback"""
        input_features = feedback_data['features']
        feedback_score = feedback_data['score']

        # Find matching patterns and update them
        for pattern in self.learning_patterns.values():
            similarity = self.calculate_similarity(
                input_features, pattern.input_features)
            if similarity > 0.7:  # Similar pattern
                # Update success rate based on feedback
                pattern.success_rate = (
                    pattern.success_rate * 0.8) + (feedback_score * 0.2)
                pattern.confidence = min(
                    pattern.confidence + (feedback_score - 0.5) * 0.1, 1.0)

    def log_feedback(self, user_input: str, ai_response: str, feedback_score: float):
        """Log feedback to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO feedback_log (input_text, output_text, feedback_score, timestamp, context_info)
            VALUES (?, ?, ?, ?, ?)
        ''', (user_input, ai_response, feedback_score, datetime.now(), json.dumps({})))

        conn.commit()
        conn.close()

    def start_background_learning(self):
        """Start background learning thread"""
        if not self.training_thread or not self.training_thread.is_alive():
            self.training_thread = threading.Thread(
                target=self.background_learning_loop)
            self.training_thread.daemon = True
            self.training_thread.start()
            logger.info("Background learning started")

    def background_learning_loop(self):
        """Background learning loop"""
        while True:
            try:
                if len(self.feedback_buffer) >= 10:  # Train when we have enough feedback
                    self.train_from_feedback()

                # Save brain state periodically
                self.save_brain_state()

                # Sleep for a while
                time.sleep(300)  # 5 minutes

            except Exception as e:
                logger.error(f"Error in background learning: {e}")
                time.sleep(60)  # Wait a minute before retrying

    def train_from_feedback(self):
        """Train the neural network from accumulated feedback"""
        if len(self.feedback_buffer) < 5:
            return

        # Prepare training data
        X = []
        y = []

        for feedback in list(self.feedback_buffer):
            features = feedback['features']
            score = feedback['score']

            # Create target vector (simplified)
            target = np.zeros(self.neural_network.output_size)
            target[int(score * (self.neural_network.output_size - 1))] = 1.0

            X.append(features)
            y.append(target)

        if len(X) > 0:
            X = np.array(X)
            y = np.array(y)

            # Train the network
            self.neural_network.train(X, y, epochs=50)
            logger.info(f"Trained on {len(X)} feedback samples")

            # Clear processed feedback
            self.feedback_buffer.clear()

    def get_learning_stats(self) -> Dict:
        """Get statistics about the AI's learning progress"""
        return {
            'total_patterns': len(self.learning_patterns),
            'avg_confidence': np.mean([p.confidence for p in self.learning_patterns.values()]) if self.learning_patterns else 0,
            'avg_success_rate': np.mean([p.success_rate for p in self.learning_patterns.values()]) if self.learning_patterns else 0,
            'knowledge_nodes': len(self.knowledge_graph.nodes),
            'feedback_samples': len(self.feedback_buffer),
            'network_accuracy': self.neural_network.accuracy_history[-1] if self.neural_network.accuracy_history else 0
        }


# Global instance
enhanced_ai_brain = SelfTrainingAI()


def get_ai_response(user_input: str, context: List[str] = None) -> str:
    """Main function to get AI response"""
    return enhanced_ai_brain.generate_response(user_input, context)


def provide_feedback(user_input: str, ai_response: str, score: float):
    """Provide feedback to improve AI responses"""
    enhanced_ai_brain.receive_feedback(user_input, ai_response, score)


def get_learning_statistics() -> Dict:
    """Get AI learning statistics"""
    return enhanced_ai_brain.get_learning_stats()
