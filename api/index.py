"""
CodeEx AI - Vercel Serverless Function
Simplified version for Vercel deployment
"""

import os
import json
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Create Flask app
app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'fallback-secret-key-for-vercel')

# Enable CORS
CORS(app, origins=['*'])

@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')

@app.route('/chat')
def chat():
    """Chat interface"""
    return render_template('chat.html')

@app.route('/api/chat', methods=['POST'])
def api_chat():
    """Simple chat API endpoint"""
    try:
        data = request.get_json()
        user_message = data.get('message', '')
        
        # Simple response for demo
        response = {
            'response': f"Echo: {user_message}",
            'status': 'success'
        }
        
        return jsonify(response)
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'}), 500

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'platform': 'vercel'})

# Vercel serverless function handler
def handler(request):
    return app(request.environ, lambda status, headers: None)

if __name__ == '__main__':
    app.run(debug=True)