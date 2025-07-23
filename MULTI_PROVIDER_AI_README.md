# 🚀 Multi-Provider AI Brain System

This system integrates **Google Gemini 1.5 Flash**, **OpenAI GPT models**, and **local AI** into a unified intelligent system with automatic fallback and load balancing.

## ✨ Features

- **Multiple AI Providers**: Google Gemini 1.5 Flash, OpenAI GPT-3.5/4, and local AI
- **Intelligent Routing**: Automatically selects the best provider based on task type
- **Automatic Fallback**: Falls back to local AI if external providers fail
- **Performance Tracking**: Monitors response times, success rates, and usage statistics
- **Task-Specific Optimization**: Different providers for coding, creative, analytical tasks
- **Caching System**: Reduces API calls with intelligent response caching
- **Feedback Learning**: Improves provider selection based on user feedback

## 🛠️ Quick Setup

### 1. Install Required Packages
```bash
pip install openai==1.3.0 google-generativeai==0.3.0
```

### 2. Get API Keys
- **OpenAI**: https://platform.openai.com/api-keys
- **Google Gemini**: https://makersuite.google.com/app/apikey

### 3. Configure Environment
Add to your `.env` file:
```env
OPENAI_API_KEY=your-openai-api-key-here
GOOGLE_GEMINI_API_KEY=your-google-gemini-api-key-here
```

### 4. Run Setup Script
```bash
python setup_ai_providers.py
```

### 5. Test the System
```bash
python test_multi_provider_ai.py
```

## 🎯 Provider Selection Strategy

The system automatically selects the best provider based on task type:

| Task Type | Primary | Secondary | Fallback |
|-----------|---------|-----------|----------|
| **Coding** | OpenAI | Gemini | Local |
| **Creative** | Gemini | OpenAI | Local |
| **Analytical** | OpenAI | Gemini | Local |
| **General** | Gemini | OpenAI | Local |
| **Fast** | Gemini | Local | OpenAI |

## 📡 API Endpoints

### Chat with Multi-Provider AI
```http
POST /api/chat/{session_id}/messages
Content-Type: application/json

{
  "message": "Write a Python function to calculate fibonacci numbers",
  "provider": "gemini",  // optional: "openai", "gemini", "local"
  "encrypt": true
}
```

### Get Available Providers
```http
GET /api/ai/providers
```

### Submit Feedback
```http
POST /api/ai/feedback
Content-Type: application/json

{
  "user_input": "Hello",
  "ai_response": "Hi there!",
  "feedback_type": "thumbs",
  "feedback_value": true,
  "provider": "gemini"
}
```

### Get AI Statistics
```http
GET /api/ai/stats
```

### Test All Providers
```http
POST /api/ai/test
Content-Type: application/json

{
  "message": "Hello! This is a test message."
}
```

## 🔧 Usage Examples

### Basic Usage
```python
from multi_provider_ai_brain import get_multi_provider_response

# Automatic provider selection
response = await get_multi_provider_response(
    "Write a Python function to sort a list",
    task_type='coding'
)

print(f"Response: {response['response']}")
print(f"Provider: {response['provider']}")
print(f"Confidence: {response['confidence']}")
```

### Specific Provider
```python
# Force specific provider
response = await get_multi_provider_response(
    "Write a creative story about AI",
    provider='gemini',
    task_type='creative'
)
```

### With Context
```python
# Include conversation context
context = ["Previous message 1", "Previous message 2"]
response = await get_multi_provider_response(
    "Continue our discussion",
    context=context
)
```

## 📊 Response Format

```json
{
  "response": "AI generated response text",
  "confidence": 0.85,
  "provider": "gemini",
  "model": "gemini-1.5-flash",
  "tokens_used": 150,
  "response_time": 1.23,
  "task_type": "coding",
  "local_analysis": {
    "intent": "programming",
    "sentiment": "neutral",
    "related_entities": []
  },
  "metadata": {
    "finish_reason": "completed",
    "safety_ratings": []
  },
  "timestamp": "2024-01-15T10:30:00"
}
```

## 🎛️ Configuration Options

### Task Type Preferences
Modify `task_preferences` in `AIProviderManager.get_best_provider()`:

```python
task_preferences = {
    'creative': ['gemini', 'openai', 'local'],
    'analytical': ['openai', 'gemini', 'local'],
    'coding': ['openai', 'gemini', 'local'],
    'general': ['gemini', 'openai', 'local'],
    'fast': ['gemini', 'local', 'openai']
}
```

### Model Selection
Default models can be changed in the provider initialization:

```python
# OpenAI models
'models': ['gpt-3.5-turbo', 'gpt-4', 'gpt-4-turbo-preview']

# Gemini models  
'models': ['gemini-1.5-flash', 'gemini-1.5-pro', 'gemini-pro']
```

## 📈 Performance Monitoring

The system tracks:
- **Response Times**: Average response time per provider
- **Success Rates**: Percentage of successful requests
- **Error Rates**: Failed requests and error types
- **Usage Statistics**: Request counts and patterns
- **Feedback Scores**: User satisfaction ratings

Access via:
```python
from multi_provider_ai_brain import get_multi_provider_stats
stats = get_multi_provider_stats()
```

## 🔄 Fallback Strategy

1. **Primary Provider**: Selected based on task type and performance
2. **Secondary Provider**: If primary fails, try secondary
3. **Local AI**: Always available as final fallback
4. **Error Handling**: Graceful degradation with informative error messages

## 🛡️ Error Handling

The system handles:
- **API Rate Limits**: Automatic retry with exponential backoff
- **Network Errors**: Fallback to alternative providers
- **Invalid API Keys**: Clear error messages and fallback
- **Model Unavailability**: Automatic model switching
- **Timeout Errors**: Configurable timeout with fallback

## 🔍 Debugging

Enable debug logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

Check provider status:
```python
from multi_provider_ai_brain import multi_provider_ai_brain
print(multi_provider_ai_brain.provider_manager.get_provider_stats())
```

## 🚀 Advanced Features

### Custom Provider Integration
Add new AI providers by extending `AIProviderManager`:

```python
async def generate_response_custom(self, prompt: str) -> AIProviderResponse:
    # Implement custom provider logic
    pass
```

### Response Caching
Responses are cached for 5 minutes by default. Modify in `MultiProviderAIBrain`:

```python
if time.time() - cached_response['timestamp'] < 300:  # 5 minutes
```

### Feedback Learning
The system learns from user feedback to improve provider selection:

```python
provide_multi_provider_feedback(
    user_input="Hello",
    ai_response="Hi there!",
    feedback_type="thumbs",
    feedback_value=True,
    provider="gemini"
)
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 🆘 Support

For issues and questions:
1. Check the test script: `python test_multi_provider_ai.py`
2. Review logs for error messages
3. Verify API key configuration
4. Test individual providers

---

**Happy coding with Multi-Provider AI! 🚀**