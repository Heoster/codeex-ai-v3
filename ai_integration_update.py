"""
AI Integration Update - Connects Flask app with Multi-Provider AI Brain
"""

import asyncio
import json
from typing import Dict, Any, List
from multi_provider_ai_brain import get_multi_provider_response, provide_multi_provider_feedback, get_multi_provider_stats

class AIIntegrationService:
    """Service to integrate multi-provider AI with Flask app"""
    
    def __init__(self):
        self.loop = None
        self.setup_event_loop()
    
    def setup_event_loop(self):
        """Setup event loop for async operations"""
        try:
            self.loop = asyncio.get_event_loop()
        except RuntimeError:
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
    
    def get_ai_response(self, user_input: str, context: List[str] = None, 
                       provider: str = None, task_type: str = 'general') -> Dict[str, Any]:
        """Synchronous wrapper for getting AI response"""
        try:
            if self.loop.is_running():
                # If loop is already running, create a new task
                future = asyncio.ensure_future(
                    get_multi_provider_response(user_input, context, provider, task_type)
                )
                # Wait for completion
                while not future.done():
                    pass
                return future.result()
            else:
                # Run in the event loop
                return self.loop.run_until_complete(
                    get_multi_provider_response(user_input, context, provider, task_type)
                )
        except Exception as e:
            # Fallback to basic response
            return {
                'response': f"I'm processing your request: {user_input}",
                'confidence': 0.5,
                'provider': 'fallback',
                'model': 'basic',
                'error': str(e)
            }
    
    def provide_feedback(self, user_input: str, ai_response: str, 
                        feedback_type: str, feedback_value: Any, provider: str = None) -> bool:
        """Provide feedback to the AI system"""
        try:
            return provide_multi_provider_feedback(
                user_input, ai_response, feedback_type, feedback_value, provider
            )
        except Exception as e:
            print(f"Error providing feedback: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get AI system statistics"""
        try:
            return get_multi_provider_stats()
        except Exception as e:
            return {'error': str(e)}
    
    def determine_task_type(self, user_input: str) -> str:
        """Determine the task type based on user input"""
        user_input_lower = user_input.lower()
        
        if any(word in user_input_lower for word in ['code', 'program', 'function', 'python', 'javascript', 'html', 'css']):
            return 'coding'
        elif any(word in user_input_lower for word in ['analyze', 'calculate', 'solve', 'math', 'statistics']):
            return 'analytical'
        elif any(word in user_input_lower for word in ['write', 'create', 'story', 'poem', 'creative', 'imagine']):
            return 'creative'
        elif any(word in user_input_lower for word in ['quick', 'fast', 'brief', 'short']):
            return 'fast'
        else:
            return 'general'

# Global service instance
ai_service = AIIntegrationService()

# Flask route helper functions
def handle_chat_request(user_input: str, context: List[str] = None, 
                       provider_preference: str = None) -> Dict[str, Any]:
    """Handle chat request from Flask app"""
    
    # Determine task type
    task_type = ai_service.determine_task_type(user_input)
    
    # Get AI response
    response = ai_service.get_ai_response(
        user_input, context, provider_preference, task_type
    )
    
    # Format response for Flask
    return {
        'success': True,
        'response': response['response'],
        'confidence': response['confidence'],
        'provider': response['provider'],
        'model': response['model'],
        'task_type': task_type,
        'metadata': {
            'tokens_used': response.get('tokens_used', 0),
            'response_time': response.get('response_time', 0),
            'local_analysis': response.get('local_analysis', {}),
            'timestamp': response.get('timestamp', '')
        }
    }

def handle_feedback_request(user_input: str, ai_response: str, 
                          feedback_type: str, feedback_value: Any, 
                          provider: str = None) -> Dict[str, Any]:
    """Handle feedback request from Flask app"""
    
    success = ai_service.provide_feedback(
        user_input, ai_response, feedback_type, feedback_value, provider
    )
    
    return {
        'success': success,
        'message': 'Feedback received successfully' if success else 'Failed to process feedback'
    }

def handle_stats_request() -> Dict[str, Any]:
    """Handle statistics request from Flask app"""
    
    stats = ai_service.get_stats()
    
    return {
        'success': True,
        'stats': stats
    }

# Example Flask route implementations
def create_flask_routes(app):
    """Create Flask routes for AI integration"""
    
    @app.route('/api/chat', methods=['POST'])
    def api_chat():
        from flask import request, jsonify
        
        data = request.get_json()
        user_input = data.get('message', '')
        context = data.get('context', [])
        provider = data.get('provider')
        
        if not user_input:
            return jsonify({'success': False, 'error': 'No message provided'}), 400
        
        response = handle_chat_request(user_input, context, provider)
        return jsonify(response)
    
    @app.route('/api/feedback', methods=['POST'])
    def api_feedback():
        from flask import request, jsonify
        
        data = request.get_json()
        user_input = data.get('user_input', '')
        ai_response = data.get('ai_response', '')
        feedback_type = data.get('feedback_type', 'thumbs')
        feedback_value = data.get('feedback_value', True)
        provider = data.get('provider')
        
        response = handle_feedback_request(
            user_input, ai_response, feedback_type, feedback_value, provider
        )
        return jsonify(response)
    
    @app.route('/api/stats', methods=['GET'])
    def api_stats():
        from flask import jsonify
        
        response = handle_stats_request()
        return jsonify(response)
    
    @app.route('/api/providers', methods=['GET'])
    def api_providers():
        from flask import jsonify
        
        stats = ai_service.get_stats()
        providers = stats.get('multi_provider_stats', {}).get('available_models', {})
        
        return jsonify({
            'success': True,
            'providers': list(providers.keys()),
            'models': providers
        })

if __name__ == "__main__":
    # Test the integration
    print("🧪 Testing AI Integration Service")
    
    # Test basic chat
    response = handle_chat_request("Hello! Can you help me with Python programming?")
    print(f"Chat Response: {json.dumps(response, indent=2)}")
    
    # Test feedback
    feedback_response = handle_feedback_request(
        "Hello! Can you help me with Python programming?",
        response['response'],
        'thumbs',
        True,
        response['provider']
    )
    print(f"Feedback Response: {json.dumps(feedback_response, indent=2)}")
    
    # Test stats
    stats_response = handle_stats_request()
    print(f"Stats Response: {json.dumps(stats_response, indent=2, default=str)}")