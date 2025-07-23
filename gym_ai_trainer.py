"""
Gymnasium AI Trainer - Reinforcement Learning for AI Brain
Trains the AI using various Gym environments to improve decision-making
"""

import gymnasium as gym
import numpy as np
import logging
import json
import sqlite3
from datetime import datetime
from typing import Dict, List, Any, Optional
from collections import deque
import os

logger = logging.getLogger(__name__)

class ConversationEnv:
    """Gym-like environment for training conversational AI"""
    
    def __init__(self):
        self.state_size = 20
        self.action_size = 8  # Number of response strategies
        self.current_state = np.zeros(self.state_size)
        self.conversation_history = []
        self.user_types = ['beginner', 'expert', 'impatient', 'curious', 'intermediate']
        self.current_user_type = 'intermediate'
        
        # Response strategies
        self.response_strategies = {
            0: 'direct_answer',
            1: 'ask_clarification',
            2: 'provide_example',
            3: 'give_detailed_explanation',
            4: 'suggest_alternative',
            5: 'empathetic_response',
            6: 'technical_deep_dive',
            7: 'question_back'
        }
        
        self.reset()
    
    def reset(self):
        """Reset environment for new conversation"""
        self.current_state = np.zeros(self.state_size)
        self.conversation_history = []
        self.current_user_type = np.random.choice(self.user_types)
        return self._get_state()
    
    def step(self, action: int):
        """Take action and return new state, reward, done"""
        strategy = self.response_strategies.get(action, 'direct_answer')
        
        # Calculate reward based on strategy and user type
        reward = self._calculate_reward(strategy)
        
        # Simulate user satisfaction (0-1)
        user_satisfaction = max(0, min(1, reward + np.random.normal(0, 0.1)))
        
        # Update conversation history
        self.conversation_history.append({
            'strategy': strategy,
            'reward': reward,
            'user_satisfaction': user_satisfaction,
            'user_type': self.current_user_type,
            'timestamp': datetime.now()
        })
        
        # Update state
        self.current_state = self._get_state()
        
        # Episode ends after 10 exchanges or if satisfaction is very low
        done = len(self.conversation_history) >= 10 or user_satisfaction < 0.2
        
        return self.current_state, reward, done, {}
    
    def _get_state(self) -> np.ndarray:
        """Get current state representation"""
        state = np.zeros(self.state_size)
        
        # Conversation length
        state[0] = len(self.conversation_history) / 10.0  # Normalized
        
        # Current user satisfaction (if available)
        if self.conversation_history:
            state[1] = self.conversation_history[-1]['user_satisfaction']
            state[2] = np.mean([h['user_satisfaction'] for h in self.conversation_history])
        
        # User type encoding (one-hot)
        user_type_idx = self.user_types.index(self.current_user_type)
        state[3 + user_type_idx] = 1.0
        
        # Recent strategy effectiveness
        if len(self.conversation_history) > 0:
            recent_rewards = [h['reward'] for h in self.conversation_history[-5:]]
            state[8] = np.mean(recent_rewards)  # Average recent reward
            state[9] = np.std(recent_rewards) if len(recent_rewards) > 1 else 0  # Reward variance
        
        # Strategy usage patterns
        if len(self.conversation_history) > 0:
            strategies_used = [h['strategy'] for h in self.conversation_history]
            for i, strategy in enumerate(self.response_strategies.values()):
                if i < 8:  # Ensure we don't exceed state size
                    state[10 + i] = strategies_used.count(strategy) / len(strategies_used)
        
        # Conversation trend
        if len(self.conversation_history) >= 3:
            recent_satisfaction = [h['user_satisfaction'] for h in self.conversation_history[-3:]]
            state[18] = recent_satisfaction[-1] - recent_satisfaction[0]  # Satisfaction trend
        
        # Random noise for exploration
        state[19] = np.random.random() * 0.1
        
        return state
    
    def _calculate_reward(self, strategy: str) -> float:
        """Calculate reward based on strategy and user type"""
        base_reward = 0.0
        
        # User type preferences
        if self.current_user_type == 'beginner':
            if strategy in ['provide_example', 'give_detailed_explanation', 'empathetic_response']:
                base_reward = 0.8
            elif strategy in ['technical_deep_dive', 'direct_answer']:
                base_reward = 0.2
            else:
                base_reward = 0.5
                
        elif self.current_user_type == 'expert':
            if strategy in ['technical_deep_dive', 'direct_answer', 'suggest_alternative']:
                base_reward = 0.9
            elif strategy in ['provide_example', 'empathetic_response']:
                base_reward = 0.3
            else:
                base_reward = 0.6
                
        elif self.current_user_type == 'impatient':
            if strategy in ['direct_answer', 'provide_example']:
                base_reward = 0.8
            elif strategy in ['question_back', 'ask_clarification', 'give_detailed_explanation']:
                base_reward = 0.1
            else:
                base_reward = 0.4
                
        elif self.current_user_type == 'curious':
            if strategy in ['give_detailed_explanation', 'technical_deep_dive', 'suggest_alternative']:
                base_reward = 0.9
            elif strategy in ['direct_answer']:
                base_reward = 0.4
            else:
                base_reward = 0.7
                
        else:  # intermediate
            base_reward = 0.6  # Neutral for all strategies
        
        # Add some randomness and context
        context_bonus = 0.0
        
        # Avoid repetitive strategies
        if len(self.conversation_history) > 2:
            recent_strategies = [h['strategy'] for h in self.conversation_history[-3:]]
            if recent_strategies.count(strategy) > 1:
                context_bonus -= 0.2  # Penalty for repetition
        
        # Reward conversation flow
        if len(self.conversation_history) > 0:
            last_satisfaction = self.conversation_history[-1]['user_satisfaction']
            if last_satisfaction > 0.7:
                context_bonus += 0.1  # Bonus for maintaining high satisfaction
        
        # Add some noise for realistic simulation
        noise = np.random.normal(0, 0.1)
        
        final_reward = np.clip(base_reward + context_bonus + noise, -1.0, 1.0)
        
        return final_reward


class GymAITrainer:
    """Reinforcement Learning trainer for AI Brain using Gymnasium"""
    
    def __init__(self):
        self.environments = {}
        self.training_history = []
        self.models = {}
        self.performance_metrics = {
            'total_episodes': 0,
            'total_rewards': 0,
            'avg_reward': 0,
            'best_reward': float('-inf'),
            'environments_trained': []
        }
        
        # Initialize basic environments
        self.initialize_environments()
        
    def initialize_environments(self):
        """Initialize various Gym environments for training"""
        try:
            # Classic control environments
            self.environments['CartPole'] = gym.make('CartPole-v1')
            self.environments['MountainCar'] = gym.make('MountainCar-v0')
            self.environments['Acrobot'] = gym.make('Acrobot-v1')
            
            logger.info("Initialized Gymnasium environments successfully")
        except Exception as e:
            logger.error(f"Error initializing environments: {e}")
            # Fallback to basic environment
            self.environments['CartPole'] = gym.make('CartPole-v1')
    
    def train_simple_agent(self, env_name: str = 'CartPole', episodes: int = 100):
        """Train a simple Q-learning agent"""
        if env_name not in self.environments:
            logger.error(f"Environment {env_name} not available")
            return None
        
        env = self.environments[env_name]
        
        # Simple Q-table for discrete environments
        if hasattr(env.observation_space, 'n'):
            # Discrete observation space
            q_table = np.zeros((env.observation_space.n, env.action_space.n))
        else:
            # Continuous observation space - use simple policy
            return self.train_simple_policy(env_name, episodes)
        
        # Training parameters
        learning_rate = 0.1
        discount_factor = 0.95
        epsilon = 1.0
        epsilon_decay = 0.995
        min_epsilon = 0.01
        
        episode_rewards = []
        
        for episode in range(episodes):
            state, _ = env.reset()
            total_reward = 0
            done = False
            
            while not done:
                # Epsilon-greedy action selection
                if np.random.random() < epsilon:
                    action = env.action_space.sample()
                else:
                    action = np.argmax(q_table[state])
                
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                # Q-learning update
                if not done:
                    q_table[state, action] += learning_rate * (
                        reward + discount_factor * np.max(q_table[next_state]) - q_table[state, action]
                    )
                else:
                    q_table[state, action] += learning_rate * (reward - q_table[state, action])
                
                state = next_state
                total_reward += reward
            
            episode_rewards.append(total_reward)
            epsilon = max(min_epsilon, epsilon * epsilon_decay)
            
            if episode % 10 == 0:
                avg_reward = np.mean(episode_rewards[-10:])
                logger.info(f"Episode {episode}, Avg Reward: {avg_reward:.2f}, Epsilon: {epsilon:.3f}")
        
        # Update performance metrics
        self.update_training_metrics(env_name, episode_rewards)
        
        return {
            'q_table': q_table,
            'episode_rewards': episode_rewards,
            'final_avg_reward': np.mean(episode_rewards[-10:])
        }
    
    def train_simple_policy(self, env_name: str, episodes: int = 100):
        """Train a simple policy for continuous environments"""
        env = self.environments[env_name]
        
        # Simple random policy with improvement
        episode_rewards = []
        best_actions = []
        
        for episode in range(episodes):
            state, _ = env.reset()
            total_reward = 0
            done = False
            actions_taken = []
            
            while not done:
                # Simple random action with learning
                action = env.action_space.sample()
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                actions_taken.append(action)
                total_reward += reward
                state = next_state
            
            episode_rewards.append(total_reward)
            
            # Learn from successful episodes
            if total_reward > np.mean(episode_rewards) if episode_rewards else 0:
                best_actions.extend(actions_taken)
            
            if episode % 10 == 0:
                avg_reward = np.mean(episode_rewards[-10:])
                logger.info(f"Episode {episode}, Avg Reward: {avg_reward:.2f}")
        
        # Update performance metrics
        self.update_training_metrics(env_name, episode_rewards)
        
        return {
            'best_actions': best_actions,
            'episode_rewards': episode_rewards,
            'final_avg_reward': np.mean(episode_rewards[-10:])
        }
    
    def update_training_metrics(self, env_name: str, episode_rewards: List[float]):
        """Update training performance metrics"""
        avg_reward = np.mean(episode_rewards)
        best_reward = max(episode_rewards)
        
        self.performance_metrics['total_episodes'] += len(episode_rewards)
        self.performance_metrics['total_rewards'] += sum(episode_rewards)
        self.performance_metrics['avg_reward'] = (
            self.performance_metrics['total_rewards'] / 
            self.performance_metrics['total_episodes']
        )
        
        if best_reward > self.performance_metrics['best_reward']:
            self.performance_metrics['best_reward'] = best_reward
        
        if env_name not in self.performance_metrics['environments_trained']:
            self.performance_metrics['environments_trained'].append(env_name)
        
        logger.info(f"Updated metrics for {env_name}: avg={avg_reward:.2f}, best={best_reward:.2f}")
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get comprehensive training summary"""
        return {
            'performance_metrics': self.performance_metrics,
            'available_environments': list(self.environments.keys()),
            'training_history': self.training_history[-10:],  # Last 10 sessions
            'models_trained': list(self.models.keys()),
            'total_training_time': sum(
                session.get('duration', 0) for session in self.training_history
            )
        }


class AdvancedGymTrainer:
    """Main trainer class that uses Gym environments to improve AI responses"""
    
    def __init__(self, db_path="gym_ai_training.db"):
        self.db_path = db_path
        self.models = {}
        self.training_history = []
        self.performance_metrics = {
            'total_episodes': 0,
            'average_reward': 0.0,
            'best_reward': -float('inf'),
            'training_time': 0.0,
            'model_improvements': 0
        }
        
        # Initialize database
        self.init_database()
        
        # Create environments
        self.conversation_env = ConversationEnv()
        
        # Initialize RL models
        self.init_rl_models()
        
        logger.info("Advanced Gym AI Trainer initialized successfully")
    
    def init_database(self):
        """Initialize database for training data"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS training_episodes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                episode_number INTEGER,
                total_reward REAL,
                episode_length INTEGER,
                user_type TEXT,
                strategies_used TEXT,
                final_satisfaction REAL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS model_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_name TEXT,
                accuracy REAL,
                loss REAL,
                training_time REAL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def init_rl_models(self):
        """Initialize reinforcement learning models"""
        try:
            # Simple Q-learning model for conversation strategies
            self.models['q_learning'] = {
                'q_table': np.zeros((self.conversation_env.state_size, self.conversation_env.action_size)),
                'learning_rate': 0.1,
                'discount_factor': 0.95,
                'epsilon': 1.0,
                'epsilon_decay': 0.995,
                'min_epsilon': 0.01
            }
            
            logger.info("RL models initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing RL models: {e}")
    
    def train_conversation_agent(self, episodes: int = 1000) -> Dict[str, Any]:
        """Train the conversation agent using reinforcement learning"""
        training_start = datetime.now()
        episode_rewards = []
        
        q_model = self.models['q_learning']
        
        for episode in range(episodes):
            state = self.conversation_env.reset()
            total_reward = 0
            step_count = 0
            done = False
            
            while not done and step_count < 20:  # Max 20 steps per episode
                # Convert state to discrete index (simplified)
                state_idx = self._discretize_state(state)
                
                # Epsilon-greedy action selection
                if np.random.random() < q_model['epsilon']:
                    action = np.random.randint(0, self.conversation_env.action_size)
                else:
                    action = np.argmax(q_model['q_table'][state_idx])
                
                # Take action
                next_state, reward, done, _ = self.conversation_env.step(action)
                next_state_idx = self._discretize_state(next_state)
                
                # Q-learning update
                if not done:
                    q_model['q_table'][state_idx, action] += q_model['learning_rate'] * (
                        reward + q_model['discount_factor'] * 
                        np.max(q_model['q_table'][next_state_idx]) - 
                        q_model['q_table'][state_idx, action]
                    )
                else:
                    q_model['q_table'][state_idx, action] += q_model['learning_rate'] * (
                        reward - q_model['q_table'][state_idx, action]
                    )
                
                state = next_state
                total_reward += reward
                step_count += 1
            
            episode_rewards.append(total_reward)
            
            # Decay epsilon
            q_model['epsilon'] = max(
                q_model['min_epsilon'], 
                q_model['epsilon'] * q_model['epsilon_decay']
            )
            
            # Log progress
            if episode % 100 == 0:
                avg_reward = np.mean(episode_rewards[-100:])
                logger.info(f"Episode {episode}: Avg Reward = {avg_reward:.3f}, Epsilon = {q_model['epsilon']:.3f}")
                
                # Save episode to database
                self._save_training_episode(
                    episode, total_reward, step_count, 
                    self.conversation_env.current_user_type,
                    self.conversation_env.conversation_history
                )
        
        training_time = (datetime.now() - training_start).total_seconds()
        
        # Update performance metrics
        self.performance_metrics.update({
            'total_episodes': self.performance_metrics['total_episodes'] + episodes,
            'average_reward': np.mean(episode_rewards),
            'best_reward': max(episode_rewards),
            'training_time': self.performance_metrics['training_time'] + training_time,
            'model_improvements': self.performance_metrics['model_improvements'] + 1
        })
        
        return {
            'episode_rewards': episode_rewards,
            'final_avg_reward': np.mean(episode_rewards[-100:]),
            'training_time': training_time,
            'total_episodes': episodes,
            'final_epsilon': q_model['epsilon']
        }
    
    def _discretize_state(self, state: np.ndarray, bins: int = 10) -> int:
        """Convert continuous state to discrete index"""
        # Simple discretization - can be improved
        state_sum = np.sum(state)
        return min(bins - 1, max(0, int(state_sum * bins)))
    
    def _save_training_episode(self, episode_num: int, reward: float, length: int, 
                              user_type: str, conversation_history: List[Dict]):
        """Save training episode to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            strategies_used = [h['strategy'] for h in conversation_history]
            final_satisfaction = conversation_history[-1]['user_satisfaction'] if conversation_history else 0.0
            
            cursor.execute('''
                INSERT INTO training_episodes 
                (episode_number, total_reward, episode_length, user_type, strategies_used, final_satisfaction)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (episode_num, reward, length, user_type, json.dumps(strategies_used), final_satisfaction))
            
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Error saving training episode: {e}")
    
    def get_best_strategy(self, state: np.ndarray) -> str:
        """Get best strategy for given state using trained model"""
        try:
            q_model = self.models['q_learning']
            state_idx = self._discretize_state(state)
            
            best_action = np.argmax(q_model['q_table'][state_idx])
            return self.conversation_env.response_strategies[best_action]
        except Exception as e:
            logger.error(f"Error getting best strategy: {e}")
            return 'direct_answer'  # Fallback


# Global trainer instance
gym_trainer = GymAITrainer()
advanced_trainer = AdvancedGymTrainer()

def train_ai_with_gym(episodes: int = 1000) -> Dict[str, Any]:
    """Train AI using gym environments"""
    return advanced_trainer.train_conversation_agent(episodes)

def get_ai_strategy_recommendation(conversation_context: List[str]) -> str:
    """Get strategy recommendation based on conversation context"""
    # Convert context to state representation (simplified)
    state = np.random.random(20)  # Placeholder - should extract features from context
    return advanced_trainer.get_best_strategy(state)

def get_training_analytics() -> Dict[str, Any]:
    """Get comprehensive training analytics"""
    return {
        'performance_metrics': advanced_trainer.performance_metrics,
        'basic_trainer_summary': gym_trainer.get_training_summary()
    }