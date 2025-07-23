"""
LLM Service - Enhanced Language Model Integration
Supports both local models (Hugging Face) and API-based models (OpenAI, Anthropic)
"""

import os
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import json

# Try importing transformers for local models
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    logging.warning("Transformers not available. Install with: pip install transformers")

# Try importing API clients
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

logger = logging.getLogger(__name__)


class LLMService:
    """Enhanced Language Model Service"""
    
    def __init__(self):
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        self.anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')
        
        # Initialize local model pipeline
        self.local_model = None
        self.model_name = "microsoft/DialoGPT-medium"  # Good for conversations
        
        # Configuration
        self.use_local_model = True  # Prefer local models for privacy
        self.max_tokens = 150
        self.temperature = 0.7
        
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize available models"""
        if HF_AVAILABLE and self.use_local_model:
            try:
                logger.info(f"Loading local model: {self.model_name}")
                self.local_model = pipeline(
                    "text-generation",
                    model=self.model_name,
                    tokenizer=self.model_name,
                    device=-1,  # CPU
                    max_length=512,
                    do_sample=True,
                    temperature=self.temperature
                )
                logger.info("Local model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load local model: {e}")
                self.local_model = None
        
        # Set up OpenAI if available
        if OPENAI_AVAILABLE and self.openai_api_key and self.openai_api_key != "your-openai-api-key-here":
            try:
                openai.api_key = self.openai_api_key
                logger.info("OpenAI API configured")
            except Exception as e:
                logger.error(f"OpenAI setup failed: {e}")
    
    def generate_response(self, user_input: str, context: List[str] = None, model_preference: str = "auto") -> Dict[str, Any]:
        """Generate response using available LLM"""
        if not context:
            context = []
        
        # Build conversation context
        conversation_context = self._build_context(user_input, context)
        
        # Try different models in order of preference
        if model_preference == "local" or (model_preference == "auto" and self.local_model):
            return self._generate_local_response(conversation_context)
        elif model_preference == "openai" or (model_preference == "auto" and self._is_openai_available()):
            return self._generate_openai_response(conversation_context)
        else:
            return self._generate_fallback_response(user_input)
    
    def _build_context(self, user_input: str, context: List[str]) -> str:
        """Build conversation context"""
        context_str = ""
        if context:
            context_str = "\n".join(context[-3:])  # Last 3 messages
            context_str += "\n"
        
        return f"{context_str}Human: {user_input}\nAssistant:"
    
    def _generate_local_response(self, prompt: str) -> Dict[str, Any]:
        """Generate response using local Hugging Face model"""
        try:
            if not self.local_model:
                raise Exception("Local model not available")
            
            # Generate response
            result = self.local_model(
                prompt,
                max_length=len(prompt.split()) + self.max_tokens,
                num_return_sequences=1,
                pad_token_id=50256,
                do_sample=True,
                temperature=self.temperature
            )
            
            # Extract generated text
            generated_text = result[0]['generated_text']
            response = generated_text[len(prompt):].strip()
            
            # Clean up response
            response = self._clean_response(response)
            
            return {
                'response': response,
                'source': 'local_llm',
                'model': self.model_name,
                'confidence': 0.8,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Local model generation failed: {e}")
            return self._generate_fallback_response(prompt.split("Human:")[-1].split("Assistant:")[0].strip())
    
    def _generate_openai_response(self, prompt: str) -> Dict[str, Any]:
        """Generate response using OpenAI API"""
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are CodeEx AI, a helpful programming assistant."},
                    {"role": "user", "content": prompt.split("Human:")[-1].split("Assistant:")[0].strip()}
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )
            
            return {
                'response': response.choices[0].message.content.strip(),
                'source': 'openai',
                'model': 'gpt-3.5-turbo',
                'confidence': 0.9,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"OpenAI generation failed: {e}")
            return self._generate_fallback_response(prompt.split("Human:")[-1].split("Assistant:")[0].strip())
    
    def _generate_fallback_response(self, user_input: str) -> Dict[str, Any]:
        """Generate fallback response when LLMs are unavailable"""
        responses = {
            'greeting': "Hello! I'm CodeEx AI. How can I help you today?",
            'python': "Python is a versatile programming language. I can help with Python concepts, libraries, and best practices.",
            'javascript': "JavaScript is essential for web development. I can assist with JS frameworks, concepts, and coding challenges.",
            'help': "I'm here to help! You can ask me about programming, get code examples, or request assistance with various topics.",
            'default': "I'm processing your request. My AI systems are learning and improving! What specific information are you looking for?"
        }
        
        user_lower = user_input.lower()
        
        if any(word in user_lower for word in ['hello', 'hi', 'hey']):
            response = responses['greeting']
        elif 'python' in user_lower:
            response = responses['python']
        elif 'javascript' in user_lower:
            response = responses['javascript']
        elif 'help' in user_lower:
            response = responses['help']
        else:
            response = responses['default']
        
        return {
            'response': response,
            'source': 'fallback',
            'model': 'rule_based',
            'confidence': 0.6,
            'timestamp': datetime.now().isoformat()
        }
    
    def _clean_response(self, response: str) -> str:
        """Clean up generated response"""
        # Remove common artifacts
        response = response.replace("Human:", "").replace("Assistant:", "")
        
        # Split on newlines and take first meaningful response
        lines = response.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            if line and not line.startswith(('Human:', 'Assistant:', 'User:')):
                cleaned_lines.append(line)
            if len(cleaned_lines) >= 3:  # Limit response length
                break
        
        return ' '.join(cleaned_lines) if cleaned_lines else "I'm here to help! What would you like to know?"
    
    def _is_openai_available(self) -> bool:
        """Check if OpenAI is properly configured"""
        return (OPENAI_AVAILABLE and 
                self.openai_api_key and 
                self.openai_api_key != "your-openai-api-key-here")
    
    def get_model_status(self) -> Dict[str, Any]:
        """Get status of available models"""
        return {
            'local_model': {
                'available': self.local_model is not None,
                'model_name': self.model_name if self.local_model else None
            },
            'openai': {
                'available': self._is_openai_available(),
                'configured': bool(self.openai_api_key)
            },
            'huggingface_installed': HF_AVAILABLE,
            'openai_installed': OPENAI_AVAILABLE
        }


# Global instance
llm_service = LLMService()