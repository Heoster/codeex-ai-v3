"""
Enhanced AI service for CodeEx with LLM integration
"""

import logging
from typing import Dict, List, Any, Optional
from llm_service import llm_service

logger = logging.getLogger(__name__)

class AIService:
    def __init__(self):
        self.name = "CodeEx AI Service"
        self.llm = llm_service
        
    def generate_response(self, user_input: str, context: List[str] = None) -> Dict[str, Any]:
        """Generate a response to user input using LLM"""
        if not context:
            context = []
        
        try:
            # Use LLM service for response generation
            response_data = self.llm.generate_response(user_input, context)
            
            # Add CodeEx-specific enhancements
            if isinstance(response_data, dict):
                response_data['service'] = 'CodeEx AI'
                return response_data
            else:
                # Handle string responses from legacy code
                return {
                    'response': str(response_data),
                    'source': 'ai_service',
                    'service': 'CodeEx AI',
                    'confidence': 0.7
                }
                
        except Exception as e:
            logger.error(f"AI response generation failed: {e}")
            return {
                'response': "I'm experiencing some technical difficulties. Please try again in a moment.",
                'source': 'error_fallback',
                'service': 'CodeEx AI',
                'confidence': 0.5,
                'error': str(e)
            }
    
    def get_simple_response(self, user_input: str) -> str:
        """Get simple string response for backward compatibility"""
        response_data = self.generate_response(user_input)
        return response_data.get('response', 'I apologize, but I cannot process your request right now.')
    
    def get_model_status(self) -> Dict[str, Any]:
        """Get status of AI models"""
        return self.llm.get_model_status()

# Global instance
ai_service = AIService()