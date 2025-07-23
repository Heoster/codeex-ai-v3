"""
🚀 Multi-Provider Advanced AI Brain System
Integrates Google Gemini 1.5 Flash, OpenAI GPT, and local ML models
"""

# Standard library imports
import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Any, Optional, Union

# Third-party imports
import numpy as np
from dotenv import load_dotenv

# Local imports
from advanced_ai_brain import advanced_ai_brain

# Load environment variables
load_dotenv()

# AI Provider imports
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logging.warning("OpenAI package not available")

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    logging.warning("Google Generative AI package not available")

# Import our local AI brain

logger = logging.getLogger(__name__)


@dataclass
class AIProviderResponse:
    """Standardized response from AI providers"""
    text: str
    confidence: float
    provider: str
    model: str
    tokens_used: int
    response_time: float
    metadata: Dict[str, Any]


class AIProviderManager:
    """Manages multiple AI providers with fallback and load balancing"""

    def __init__(self):
        self.providers = {}
        self.provider_stats = {}
        self.initialize_providers()

    def initialize_providers(self):
        """Initialize all available AI providers"""

        # Initialize OpenAI
        if OPENAI_AVAILABLE and os.getenv('OPENAI_API_KEY'):
            try:
                openai.api_key = os.getenv('OPENAI_API_KEY')
                self.providers['openai'] = {
                    'client': openai,
                    'models': ['gpt-3.5-turbo', 'gpt-4', 'gpt-4-turbo-preview'],
                    'available': True,
                    'rate_limit': 60,  # requests per minute
                    'last_request': 0
                }
                self.provider_stats['openai'] = {
                    'requests': 0,
                    'errors': 0,
                    'avg_response_time': 0.0,
                    'success_rate': 1.0
                }
                logger.info("OpenAI provider initialized")
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI: {e}")

        # Initialize Google Gemini
        if GEMINI_AVAILABLE and os.getenv('GOOGLE_GEMINI_API_KEY'):
            try:
                genai.configure(api_key=os.getenv('GOOGLE_GEMINI_API_KEY'))
                self.providers['gemini'] = {
                    'client': genai,
                    'models': ['gemini-1.5-flash', 'gemini-1.5-pro', 'gemini-pro'],
                    'available': True,
                    'rate_limit': 60,
                    'last_request': 0
                }
                self.provider_stats['gemini'] = {
                    'requests': 0,
                    'errors': 0,
                    'avg_response_time': 0.0,
                    'success_rate': 1.0
                }
                logger.info("Google Gemini provider initialized")
            except Exception as e:
                logger.error(f"Failed to initialize Gemini: {e}")

        # Always have local AI as fallback
        self.providers['local'] = {
            'client': advanced_ai_brain,
            'models': ['advanced_local_ai'],
            'available': True,
            'rate_limit': 1000,  # No rate limit for local
            'last_request': 0
        }
        self.provider_stats['local'] = {
            'requests': 0,
            'errors': 0,
            'avg_response_time': 0.0,
            'success_rate': 1.0
        }

        logger.info(f"Initialized {len(self.providers)} AI providers")

    def get_best_provider(self, task_type: str = 'general') -> str:
        """Select the best provider based on task type and performance"""

        # Task-specific provider preferences
        task_preferences = {
            'creative': ['gemini', 'openai', 'local'],
            'analytical': ['openai', 'gemini', 'local'],
            'coding': ['openai', 'gemini', 'local'],
            'general': ['gemini', 'openai', 'local'],
            'fast': ['gemini', 'local', 'openai']
        }

        preferred_order = task_preferences.get(
            task_type, ['gemini', 'openai', 'local'])

        # Filter available providers and sort by success rate
        available_providers = []
        for provider in preferred_order:
            if provider in self.providers and self.providers[provider]['available']:
                stats = self.provider_stats[provider]
                available_providers.append((provider, stats['success_rate']))

        if not available_providers:
            return 'local'  # Always fallback to local

        # Return provider with highest success rate
        return max(available_providers, key=lambda x: x[1])[0]

    async def generate_response_openai(self, prompt: str, model: str = 'gpt-3.5-turbo') -> AIProviderResponse:
        """Generate response using OpenAI"""
        start_time = time.time()

        try:
            client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1000,
                temperature=0.7
            )

            response_time = time.time() - start_time

            return AIProviderResponse(
                text=response.choices[0].message.content,
                confidence=0.85,  # OpenAI doesn't provide confidence scores
                provider='openai',
                model=model,
                tokens_used=response.usage.total_tokens,
                response_time=response_time,
                metadata={
                    'finish_reason': response.choices[0].finish_reason,
                    'prompt_tokens': response.usage.prompt_tokens,
                    'completion_tokens': response.usage.completion_tokens
                }
            )

        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise

    async def generate_response_gemini(self, prompt: str, model: str = 'gemini-1.5-flash') -> AIProviderResponse:
        """Generate response using Google Gemini"""
        start_time = time.time()

        try:
            model_instance = genai.GenerativeModel(model)
            response = model_instance.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=1000,
                    temperature=0.7,
                )
            )

            response_time = time.time() - start_time

            # Calculate confidence based on response quality indicators
            confidence = 0.8
            if hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, 'safety_ratings'):
                    confidence = 0.9  # Higher confidence if safety ratings are good

            return AIProviderResponse(
                text=response.text,
                confidence=confidence,
                provider='gemini',
                model=model,
                tokens_used=len(prompt.split()) +
                len(response.text.split()),  # Approximate
                response_time=response_time,
                metadata={
                    'finish_reason': 'completed',
                    'safety_ratings': getattr(response.candidates[0], 'safety_ratings', []) if hasattr(response, 'candidates') else []
                }
            )

        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            raise

    async def generate_response_local(self, prompt: str) -> AIProviderResponse:
        """Generate response using local AI brain"""
        start_time = time.time()

        try:
            response = advanced_ai_brain.generate_advanced_response(prompt)
            response_time = time.time() - start_time

            return AIProviderResponse(
                text=response['response'],
                confidence=response['confidence'],
                provider='local',
                model='advanced_local_ai',
                tokens_used=len(prompt.split()) +
                len(response['response'].split()),
                response_time=response_time,
                metadata={
                    'intent': response.get('intent', 'unknown'),
                    'sentiment': response.get('sentiment', 'neutral'),
                    'source': response.get('source', 'generated')
                }
            )

        except Exception as e:
            logger.error(f"Local AI error: {e}")
            raise

    async def generate_response(self, prompt: str, provider: str = None, model: str = None,
                                task_type: str = 'general') -> AIProviderResponse:
        """Generate response using specified or best available provider"""

        if not provider:
            provider = self.get_best_provider(task_type)

        # Update request stats
        self.provider_stats[provider]['requests'] += 1

        try:
            if provider == 'openai':
                model = model or 'gpt-3.5-turbo'
                response = await self.generate_response_openai(prompt, model)
            elif provider == 'gemini':
                model = model or 'gemini-1.5-flash'
                response = await self.generate_response_gemini(prompt, model)
            elif provider == 'local':
                response = await self.generate_response_local(prompt)
            else:
                raise ValueError(f"Unknown provider: {provider}")

            # Update success stats
            stats = self.provider_stats[provider]
            stats['avg_response_time'] = (
                (stats['avg_response_time'] * (stats['requests'] - 1) + response.response_time) /
                stats['requests']
            )
            stats['success_rate'] = (
                stats['requests'] - stats['errors']) / stats['requests']

            return response

        except Exception as e:
            # Update error stats
            self.provider_stats[provider]['errors'] += 1
            stats = self.provider_stats[provider]
            stats['success_rate'] = (
                stats['requests'] - stats['errors']) / stats['requests']

            # Try fallback provider
            if provider != 'local':
                logger.warning(
                    f"Provider {provider} failed, falling back to local AI")
                return await self.generate_response(prompt, 'local', task_type=task_type)
            else:
                raise

    def get_provider_stats(self) -> Dict[str, Any]:
        """Get statistics for all providers"""
        return {
            'providers': list(self.providers.keys()),
            'stats': self.provider_stats,
            'available_models': {
                provider: info['models']
                for provider, info in self.providers.items()
                if info['available']
            }
        }


class MultiProviderAIBrain:
    """Enhanced AI Brain that uses multiple providers with intelligent routing"""

    def __init__(self):
        self.provider_manager = AIProviderManager()
        self.local_brain = advanced_ai_brain
        self.conversation_history = []
        self.response_cache = {}

        logger.info("Multi-Provider AI Brain initialized")

    async def generate_response(self, user_input: str, context: List[str] = None,
                                provider_preference: str = None, task_type: str = 'general') -> Dict[str, Any]:
        """Generate response using the best available AI provider"""

        # Check cache first
        cache_key = f"{user_input}_{task_type}"
        if cache_key in self.response_cache:
            cached_response = self.response_cache[cache_key]
            # 5 minute cache
            if time.time() - cached_response['timestamp'] < 300:
                return cached_response['response']

        try:
            # Get response from provider
            ai_response = await self.provider_manager.generate_response(
                user_input, provider_preference, task_type=task_type
            )

            # Enhance with local AI analysis
            local_analysis = self.local_brain.generate_advanced_response(
                user_input, context)

            # Combine responses
            enhanced_response = {
                'response': ai_response.text,
                'confidence': ai_response.confidence,
                'provider': ai_response.provider,
                'model': ai_response.model,
                'tokens_used': ai_response.tokens_used,
                'response_time': ai_response.response_time,
                'local_analysis': {
                    'intent': local_analysis.get('intent', 'unknown'),
                    'sentiment': local_analysis.get('sentiment', 'neutral'),
                    'related_entities': local_analysis.get('related_entities', [])
                },
                'metadata': ai_response.metadata,
                'task_type': task_type,
                'timestamp': datetime.now().isoformat()
            }

            # Cache response
            self.response_cache[cache_key] = {
                'response': enhanced_response,
                'timestamp': time.time()
            }

            # Store in conversation history
            self.conversation_history.append({
                'input': user_input,
                'response': enhanced_response,
                'timestamp': datetime.now()
            })

            return enhanced_response

        except Exception as e:
            logger.error(f"Error generating multi-provider response: {e}")
            # Fallback to local AI only
            return self.local_brain.generate_advanced_response(user_input, context)

    def provide_feedback(self, user_input: str, ai_response: str, feedback_type: str,
                         feedback_value: Any, provider: str = None) -> bool:
        """Provide feedback to improve AI responses"""
        try:
            # Always provide feedback to local brain for learning
            success = self.local_brain.receive_advanced_feedback(
                user_input, ai_response, feedback_type, feedback_value
            )

            # Update provider stats based on feedback
            if provider and provider in self.provider_manager.provider_stats:
                stats = self.provider_manager.provider_stats[provider]
                if feedback_type == 'thumbs' and feedback_value:
                    # Positive feedback - slightly increase success rate
                    stats['success_rate'] = min(
                        1.0, stats['success_rate'] * 1.01)
                elif feedback_type == 'thumbs' and not feedback_value:
                    # Negative feedback - slightly decrease success rate
                    stats['success_rate'] = max(
                        0.1, stats['success_rate'] * 0.99)

            return success

        except Exception as e:
            logger.error(f"Error providing feedback: {e}")
            return False

    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics from all AI components"""
        provider_stats = self.provider_manager.get_provider_stats()
        local_stats = self.local_brain.get_advanced_statistics()

        return {
            'multi_provider_stats': provider_stats,
            'local_ai_stats': local_stats,
            'conversation_history': len(self.conversation_history),
            'cache_size': len(self.response_cache),
            'total_providers': len(provider_stats['providers']),
            'available_models': provider_stats['available_models']
        }


# Global instance
multi_provider_ai_brain = MultiProviderAIBrain()

# Main interface functions


async def get_multi_provider_response(user_input: str, context: List[str] = None,
                                      provider: str = None, task_type: str = 'general') -> Dict[str, Any]:
    """Get AI response using multiple providers"""
    return await multi_provider_ai_brain.generate_response(user_input, context, provider, task_type)


def provide_multi_provider_feedback(user_input: str, ai_response: str, feedback_type: str,
                                    feedback_value: Any, provider: str = None) -> bool:
    """Provide feedback to multi-provider AI system"""
    return multi_provider_ai_brain.provide_feedback(user_input, ai_response, feedback_type, feedback_value, provider)


def get_multi_provider_stats() -> Dict[str, Any]:
    """Get comprehensive multi-provider statistics"""
    return multi_provider_ai_brain.get_comprehensive_stats()


if __name__ == "__main__":
    async def test_multi_provider_ai():
        print("🚀 Multi-Provider AI Brain Test")

        # Test basic response
        response = await get_multi_provider_response(
            "Hello! Can you help me write a Python function to calculate fibonacci numbers?",
            task_type='coding'
        )

        print(f"Response: {response['response']}")
        print(f"Provider: {response['provider']}")
        print(f"Model: {response['model']}")
        print(f"Confidence: {response['confidence']}")
        print(f"Response time: {response['response_time']:.3f}s")

        # Get stats
        stats = get_multi_provider_stats()
        print(f"\nStats: {json.dumps(stats, indent=2, default=str)}")

    # Run the test
    asyncio.run(test_multi_provider_ai())
