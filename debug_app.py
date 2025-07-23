#!/usr/bin/env python3
"""
Debug version of the app to isolate issues
"""

import sys
import traceback

try:
    print("Starting imports...")
    
    # Standard library imports
    import os
    import uuid
    import sqlite3
    import smtplib
    import logging
    import secrets
    import hashlib
    from datetime import datetime, timedelta
    from functools import wraps
    from email.mime.text import MIMEText
    from email.mime.multipart import MIMEMultipart
    from email.utils import formataddr
    print("✓ Standard library imports successful")

    # Third-party imports
    from flask import Flask, render_template, request, jsonify, session, redirect, url_for, flash
    from flask_cors import CORS
    from werkzeug.security import generate_password_hash, check_password_hash
    from dotenv import load_dotenv
    print("✓ Flask imports successful")

    # Load environment variables
    load_dotenv()
    print("✓ Environment variables loaded")

    # Configure logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    print("✓ Logging configured")

    # Create Flask app
    app = Flask(__name__)
    app.secret_key = os.environ.get('SECRET_KEY', 'debug-secret-key')
    print("✓ Flask app created")

    # Simple configuration for debugging
    app.config.update(
        SECRET_KEY=os.environ.get('SECRET_KEY', 'debug-secret-key'),
        SESSION_COOKIE_SECURE=False,  # Allow HTTP for debugging
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE='Lax',
    )
    print("✓ App configuration set")

    # CORS
    CORS(app)
    print("✓ CORS configured")

    @app.route('/')
    def index():
        return jsonify({
            'status': 'success',
            'message': 'CodeEx AI Debug Mode',
            'timestamp': datetime.now().isoformat()
        })

    @app.route('/health')
    def health():
        return jsonify({'health': 'ok'})

    print("✓ Routes defined")
    print("🚀 Starting debug server...")
    
    if __name__ == '__main__':
        app.run(debug=True, host='127.0.0.1', port=5002)

except Exception as e:
    print(f"❌ Error during initialization: {e}")
    print("Full traceback:")
    traceback.print_exc()
    sys.exit(1)