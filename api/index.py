"""
CodeEx AI - Vercel Serverless Function
Clean and organized Flask application for Vercel deployment
"""

import os
import sys
import json
import logging
import uuid
import re
from datetime import datetime
from flask import Flask, render_template, request, jsonify, session, redirect, url_for, flash
from flask_cors import CORS
from dotenv import load_dotenv

# AI Integration imports
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get the parent directory (project root)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Create Flask app with correct paths
app = Flask(__name__, 
           template_folder=os.path.join(project_root, 'public'),
           static_folder=os.path.join(project_root, 'public'))

# App configuration
app.secret_key = os.getenv('SECRET_KEY', 'fallback-secret-key-for-vercel-2024')
app.config['DEBUG'] = os.getenv('FLASK_DEBUG', 'false').lower() == 'true'

# Enable CORS
CORS(app, origins=['*'])

# Configure AI services
if OPENAI_AVAILABLE:
    openai_api_key = os.getenv('OPENAI_API_KEY')
    if openai_api_key:
        openai.api_key = openai_api_key
        logger.info("OpenAI configured successfully")
    else:
        OPENAI_AVAILABLE = False
        logger.warning("OpenAI API key not found")

if GEMINI_AVAILABLE:
    gemini_api_key = os.getenv('GOOGLE_GEMINI_API_KEY')
    if gemini_api_key:
        genai.configure(api_key=gemini_api_key)
        logger.info("Gemini configured successfully")
    else:
        GEMINI_AVAILABLE = False
        logger.warning("Gemini API key not found")

# ============================================================================
# AI CONFIGURATION
# ============================================================================

# Configure OpenAI
if OPENAI_AVAILABLE:
    openai_api_key = os.getenv('OPENAI_API_KEY')
    if openai_api_key:
        openai.api_key = openai_api_key
        logger.info("OpenAI configured successfully")
    else:
        logger.warning("OpenAI API key not found")
        OPENAI_AVAILABLE = False

# Configure Gemini
if GEMINI_AVAILABLE:
    gemini_api_key = os.getenv('GOOGLE_GEMINI_API_KEY')
    if gemini_api_key:
        genai.configure(api_key=gemini_api_key)
        logger.info("Gemini configured successfully")
    else:
        logger.warning("Gemini API key not found")
        GEMINI_AVAILABLE = False

# ============================================================================
# AI HELPER FUNCTIONS
# ============================================================================

def generate_message_id():
    """Generate unique message ID"""
    return str(uuid.uuid4())[:8]

def clean_message(message):
    """Clean and sanitize user message"""
    # Remove excessive whitespace
    message = re.sub(r'\s+', ' ', message.strip())
    # Remove potentially harmful content
    message = re.sub(r'<[^>]*>', '', message)  # Remove HTML tags
    return message[:1000]  # Limit length

def get_ai_response(user_message):
    """Get AI response using available providers"""
    user_message = clean_message(user_message)
    
    # Try OpenAI first
    if OPENAI_AVAILABLE:
        try:
            response = get_openai_response(user_message)
            if response:
                return response
        except Exception as e:
            logger.error(f"OpenAI error: {e}")
    
    # Try Gemini as fallback
    if GEMINI_AVAILABLE:
        try:
            response = get_gemini_response(user_message)
            if response:
                return response
        except Exception as e:
            logger.error(f"Gemini error: {e}")
    
    # Return intelligent fallback
    return get_intelligent_fallback(user_message)

def get_openai_response(message):
    """Get response from OpenAI"""
    if not OPENAI_AVAILABLE:
        return None
    
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are CodeEx AI, a helpful programming and general knowledge assistant. Be concise, accurate, and friendly."},
                {"role": "user", "content": message}
            ],
            max_tokens=500,
            temperature=0.7
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"OpenAI API error: {e}")
        return None

def get_gemini_response(message):
    """Get response from Google Gemini"""
    if not GEMINI_AVAILABLE:
        return None
    
    try:
        model = genai.GenerativeModel('gemini-pro')
        prompt = f"You are CodeEx AI, a helpful programming and general knowledge assistant. Be concise, accurate, and friendly.\n\nUser: {message}\nCodeEx AI:"
        
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        logger.error(f"Gemini API error: {e}")
        return None

def get_intelligent_fallback(message):
    """Provide intelligent fallback responses"""
    message_lower = message.lower()
    
    # Programming related
    if any(word in message_lower for word in ['code', 'programming', 'python', 'javascript', 'html', 'css', 'bug', 'error']):
        return f"I'd be happy to help with your programming question about: '{message}'. However, I'm currently running in offline mode. For the best coding assistance, please ensure your API keys are configured."
    
    # General questions
    elif any(word in message_lower for word in ['what', 'how', 'why', 'when', 'where']):
        return f"That's an interesting question about '{message}'. I'm currently in offline mode, but I can still provide basic assistance. Could you be more specific about what you'd like to know?"
    
    # Greetings
    elif any(word in message_lower for word in ['hello', 'hi', 'hey', 'greetings']):
        return "Hello! I'm CodeEx AI, your programming and general knowledge assistant. How can I help you today? (Note: I'm currently in offline mode - configure API keys for full functionality)"
    
    # Default response
    else:
        return f"I understand you're asking about: '{message}'. I'm CodeEx AI and I'm here to help! Currently running in offline mode - please configure OpenAI or Gemini API keys for enhanced responses."

def get_fallback_response(message):
    """Get fallback response when AI services fail"""
    return f"I apologize, but I'm experiencing technical difficulties. Your message '{message}' was received, but I cannot process it right now. Please try again later."

def log_conversation(user_message, ai_response):
    """Log conversation for analytics"""
    try:
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'user_message': user_message[:100],  # Truncate for privacy
            'response_length': len(ai_response),
            'ai_provider': 'openai' if OPENAI_AVAILABLE else 'gemini' if GEMINI_AVAILABLE else 'fallback'
        }
        logger.info(f"Conversation logged: {log_entry}")
    except Exception as e:
        logger.error(f"Logging error: {e}")

# ============================================================================
# MAIN ROUTES
# ============================================================================

@app.route('/')
def index():
    """Main landing page"""
    try:
        return render_template('index.html')
    except Exception as e:
        logger.error(f"Error rendering index: {e}")
        return jsonify({'error': 'Template not found', 'status': 'error'}), 500

@app.route('/dashboard')
def dashboard():
    """AI Dashboard page"""
    try:
        return render_template('ai_dashboard.html')
    except Exception as e:
        logger.error(f"Error rendering dashboard: {e}")
        return jsonify({'error': 'Dashboard template not found', 'status': 'error'}), 500

@app.route('/chat')
def chat():
    """Chat interface page"""
    try:
        return render_template('chat.html')
    except Exception as e:
        logger.error(f"Error rendering chat: {e}")
        return jsonify({'error': 'Chat template not found', 'status': 'error'}), 500

@app.route('/login', methods=['GET', 'POST'])
def login():
    """Login page"""
    try:
        if request.method == 'POST':
            # Handle login form submission
            data = request.get_json() or request.form
            username = data.get('username', '')
            password = data.get('password', '')
            
            # Simple demo authentication (replace with real auth)
            if username and password:
                return jsonify({
                    'status': 'success',
                    'message': 'Login successful (demo)',
                    'redirect': '/dashboard'
                })
            else:
                return jsonify({
                    'status': 'error',
                    'message': 'Invalid credentials'
                }), 400
        
        return render_template('login.html')
    except Exception as e:
        logger.error(f"Error in login: {e}")
        return jsonify({'error': 'Login error', 'status': 'error'}), 500

@app.route('/contact')
def contact():
    """Contact page"""
    try:
        return render_template('contact.html')
    except Exception as e:
        logger.error(f"Error rendering contact: {e}")
        return jsonify({'error': 'Contact template not found', 'status': 'error'}), 500

@app.route('/faq')
def faq():
    """FAQ page"""
    try:
        return render_template('faq.html')
    except Exception as e:
        logger.error(f"Error rendering FAQ: {e}")
        return jsonify({'error': 'FAQ template not found', 'status': 'error'}), 500

@app.route('/about')
def about():
    """About page"""
    try:
        return render_template('about.html')
    except Exception as e:
        logger.error(f"Error rendering about: {e}")
        return jsonify({'error': 'About template not found', 'status': 'error'}), 500

@app.route('/docs')
def docs():
    """Documentation page"""
    try:
        return render_template('docs.html')
    except Exception as e:
        logger.error(f"Error rendering docs: {e}")
        return jsonify({'error': 'Docs template not found', 'status': 'error'}), 500

@app.route('/privacy')
def privacy():
    """Privacy policy page"""
    try:
        return render_template('policy.html')
    except Exception as e:
        logger.error(f"Error rendering privacy: {e}")
        return jsonify({'error': 'Privacy template not found', 'status': 'error'}), 500

@app.route('/ai-brain-dashboard')
def ai_brain_dashboard():
    """AI Brain Dashboard page"""
    try:
        return render_template('ai_dashboard.html')
    except Exception as e:
        logger.error(f"Error rendering AI dashboard: {e}")
        return jsonify({'error': 'AI Dashboard template not found', 'status': 'error'}), 500

@app.route('/forgot-password')
def forgot_password():
    """Forgot password page"""
    try:
        return render_template('forgot_password.html')
    except Exception as e:
        logger.error(f"Error rendering forgot password: {e}")
        return jsonify({'error': 'Forgot password template not found', 'status': 'error'}), 500

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.route('/api/chat', methods=['POST'])
def api_chat():
    """Advanced AI Chat API endpoint"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided', 'status': 'error'}), 400
            
        user_message = data.get('message', '').strip()
        if not user_message:
            return jsonify({'error': 'Empty message', 'status': 'error'}), 400
        
        # Get AI response using multiple providers
        ai_response = get_ai_response(user_message)
        
        response = {
            'response': ai_response,
            'status': 'success',
            'timestamp': datetime.now().isoformat(),
            'message_id': generate_message_id(),
            'user_message': user_message
        }
        
        # Log the conversation
        log_conversation(user_message, ai_response)
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Chat API error: {e}")
        return jsonify({
            'error': 'AI service temporarily unavailable',
            'status': 'error',
            'fallback_response': get_fallback_response(user_message)
        }), 500

@app.route('/api/health', methods=['GET'])
def api_health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'platform': 'vercel',
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0'
    })

@app.route('/api/status', methods=['GET'])
def api_status():
    """System status endpoint"""
    return jsonify({
        'app': 'CodeEx AI',
        'status': 'running',
        'platform': 'vercel-serverless',
        'python_version': sys.version,
        'ai_providers': {
            'openai': OPENAI_AVAILABLE,
            'gemini': GEMINI_AVAILABLE
        },
        'timestamp': datetime.now().isoformat()
    })

@app.route('/favicon.ico')
def favicon():
    """Favicon route"""
    try:
        return app.send_static_file('images/favicon-32x32.png')
    except:
        return '', 404

@app.route('/static/<path:filename>')
def static_files(filename):
    """Handle static file requests - redirect to public folder"""
    try:
        return app.send_static_file(filename)
    except:
        return '', 404

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def generate_message_id():
    """Generate unique message ID"""
    return str(uuid.uuid4())

def log_conversation(user_message, ai_response):
    """Log conversation for debugging"""
    logger.info(f"User: {user_message[:100]}...")
    logger.info(f"AI: {ai_response[:100]}...")

def get_ai_response(message):
    """Get AI response from available providers"""
    try:
        # Try OpenAI first
        if OPENAI_AVAILABLE:
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are CodeEx AI, a helpful programming assistant."},
                        {"role": "user", "content": message}
                    ],
                    max_tokens=500,
                    temperature=0.7
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                logger.error(f"OpenAI error: {e}")
        
        # Try Gemini as fallback
        if GEMINI_AVAILABLE:
            try:
                model = genai.GenerativeModel('gemini-pro')
                response = model.generate_content(f"As CodeEx AI, a programming assistant: {message}")
                return response.text.strip()
            except Exception as e:
                logger.error(f"Gemini error: {e}")
        
        # Fallback response
        return f"CodeEx AI: I received your message about '{message}'. I'm currently in demo mode. Please configure your AI API keys for full functionality."
        
    except Exception as e:
        logger.error(f"AI response error: {e}")
        return "CodeEx AI: I'm experiencing technical difficulties. Please try again later."

@app.route('/api/ai-info', methods=['GET'])
def api_ai_info():
    """AI service information"""
    return jsonify({
        'ai_providers': {
            'openai': {
                'available': OPENAI_AVAILABLE,
                'model': 'gpt-3.5-turbo' if OPENAI_AVAILABLE else None
            },
            'gemini': {
                'available': GEMINI_AVAILABLE,
                'model': 'gemini-pro' if GEMINI_AVAILABLE else None
            }
        },
        'features': [
            'Programming assistance',
            'General knowledge Q&A',
            'Code debugging help',
            'Technical explanations',
            'Multi-provider fallback'
        ],
        'status': 'operational' if (OPENAI_AVAILABLE or GEMINI_AVAILABLE) else 'offline-mode'
    })

@app.route('/api/test-ai', methods=['POST'])
def api_test_ai():
    """Test AI functionality"""
    try:
        test_message = "Hello, can you help me with programming?"
        response = get_ai_response(test_message)
        
        return jsonify({
            'test_message': test_message,
            'ai_response': response,
            'providers_tested': {
                'openai': OPENAI_AVAILABLE,
                'gemini': GEMINI_AVAILABLE
            },
            'status': 'success'
        })
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500

# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    try:
        return render_template('404.html'), 404
    except:
        return jsonify({'error': 'Page not found', 'status': 'error'}), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    try:
        return render_template('500.html'), 500
    except:
        return jsonify({'error': 'Internal server error', 'status': 'error'}), 500

@app.errorhandler(403)
def forbidden(error):
    """Handle 403 errors"""
    try:
        return render_template('403.html'), 403
    except:
        return jsonify({'error': 'Forbidden', 'status': 'error'}), 403

# ============================================================================
# VERCEL HANDLER
# ============================================================================

def handler(event, context):
    """Vercel serverless function handler"""
    try:
        return app(event, context)
    except Exception as e:
        logger.error(f"Handler error: {e}")
        return {
            'statusCode': 500,
            'body': json.dumps({'error': 'Server error', 'status': 'error'})
        }

# ============================================================================
# DEVELOPMENT SERVER
# ============================================================================

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    debug = os.getenv('FLASK_DEBUG', 'false').lower() == 'true'
    
    logger.info(f"Starting CodeEx AI on port {port}")
    logger.info(f"Debug mode: {debug}")
    logger.info(f"Template folder: {app.template_folder}")
    logger.info(f"Static folder: {app.static_folder}")
    
    app.run(host='0.0.0.0', port=port, debug=debug)