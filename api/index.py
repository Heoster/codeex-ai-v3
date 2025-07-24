"""
CodeEx AI - Vercel Serverless Function
Clean and organized Flask application for Vercel deployment
"""

import os
import sys
import json
import logging
from datetime import datetime
from flask import Flask, render_template, request, jsonify, session, redirect, url_for, flash
from flask_cors import CORS
from dotenv import load_dotenv

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

@app.route('/login')
def login():
    """Login page"""
    try:
        return render_template('login.html')
    except Exception as e:
        logger.error(f"Error rendering login: {e}")
        return jsonify({'error': 'Login template not found', 'status': 'error'}), 500

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

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.route('/api/chat', methods=['POST'])
def api_chat():
    """Chat API endpoint"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided', 'status': 'error'}), 400
            
        user_message = data.get('message', '').strip()
        if not user_message:
            return jsonify({'error': 'Empty message', 'status': 'error'}), 400
        
        # Simple echo response for demo (replace with actual AI logic)
        response = {
            'response': f"CodeEx AI: I received your message: '{user_message}'. This is a demo response.",
            'status': 'success',
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Chat API error: {e}")
        return jsonify({'error': str(e), 'status': 'error'}), 500

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
        'timestamp': datetime.now().isoformat()
    })

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