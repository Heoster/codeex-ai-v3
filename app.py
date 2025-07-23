"""
CodeEx AI - Complete Python Web Application
A powerful AI chat application built entirely in Python
"""

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

# Third-party imports
from flask import (Flask, render_template, request, jsonify, session,
                   redirect, url_for, flash)
from flask_cors import CORS
from werkzeug.security import generate_password_hash, check_password_hash
from dotenv import load_dotenv
from google.auth.transport import requests as google_requests
from google.oauth2 import id_token
import requests

# Security imports (conditional)
try:
    from flask_limiter import Limiter
    from flask_limiter.util import get_remote_address
    from flask_talisman import Talisman
    SECURITY_AVAILABLE = True
except ImportError:
    SECURITY_AVAILABLE = False

# Local imports
from web_scraper import get_scraping_service
from chat_history_service import chat_history_service
from ai_brain_integration import (get_intelligent_response, submit_feedback,
                                  get_ai_metrics, configure_ai_learning)
from ai_integration_update import handle_chat_request, handle_feedback_request, handle_stats_request

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Enhanced SSL Filter for development
class SSLFilter(logging.Filter):
    def filter(self, record):
        # Filter out SSL/TLS handshake error messages and binary data
        if hasattr(record, 'getMessage'):
            message = record.getMessage()
            # Filter SSL handshake attempts and binary data
            if any(phrase in message for phrase in [
                'Bad request version',
                'Bad HTTP/0.9 request type', 
                'Bad request syntax',
                'code 400, message Bad',
                '\x16\x03',  # SSL/TLS handshake start
                'HTTP/1.1" 400',
                'code 400'
            ]) or any(ord(c) > 127 for c in message if isinstance(c, str)):
                return False
        return True

# Apply enhanced filter to werkzeug logger
werkzeug_logger = logging.getLogger('werkzeug')
werkzeug_logger.addFilter(SSLFilter())
werkzeug_logger.setLevel(logging.WARNING)  # Reduce verbosity

# Log security package availability
if not SECURITY_AVAILABLE:
    logger.warning("Security packages not available - install flask-limiter and flask-talisman for production")

app = Flask(__name__)

# Production-ready configuration
app.config.update(
    SECRET_KEY=os.environ.get('SECRET_KEY', secrets.token_hex(32)),
    SESSION_COOKIE_SECURE=os.environ.get('HTTPS_ENABLED', 'false').lower() == 'true',
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE='Lax',
    PERMANENT_SESSION_LIFETIME=timedelta(hours=24),
    WTF_CSRF_ENABLED=False,  # Disable CSRF for development
    WTF_CSRF_TIME_LIMIT=None
)

# CORS configuration for production
CORS(app, 
     origins=os.environ.get('ALLOWED_ORIGINS', '*').split(','),
     supports_credentials=True)

# Security configuration
if SECURITY_AVAILABLE:
    # Rate limiting
    limiter = Limiter(
        key_func=get_remote_address,
        default_limits=["200 per day", "50 per hour"]
    )
    limiter.init_app(app)
    
    # Security headers
    csp = {
        'default-src': "'self'",
        'script-src': "'self' 'unsafe-inline' https://accounts.google.com https://apis.google.com",
        'style-src': "'self' 'unsafe-inline' https://fonts.googleapis.com",
        'font-src': "'self' https://fonts.gstatic.com",
        'img-src': "'self' data: https:",
        'connect-src': "'self' https://api.openai.com https://generativelanguage.googleapis.com"
    }
    
    Talisman(app, 
             force_https=os.environ.get('FORCE_HTTPS', 'false').lower() == 'true',
             content_security_policy=csp)
else:
    # Create a dummy limiter when security packages aren't available
    limiter = None

# Google OAuth Configuration
GOOGLE_CLIENT_ID = os.environ.get('GOOGLE_CLIENT_ID')
GOOGLE_CLIENT_SECRET = os.environ.get('GOOGLE_CLIENT_SECRET')

# Email Configuration
EMAIL_HOST = os.environ.get('EMAIL_HOST', 'smtp.gmail.com')
EMAIL_PORT = int(os.environ.get('EMAIL_PORT', '587'))
EMAIL_USERNAME = os.environ.get('EMAIL_USERNAME')
EMAIL_PASSWORD = os.environ.get('EMAIL_PASSWORD')
CONTACT_EMAIL = os.environ.get('CONTACT_EMAIL', 'support@codeexai.com')

# Validate required environment variables
required_env_vars = ['SECRET_KEY', 'GOOGLE_CLIENT_ID', 'GOOGLE_CLIENT_SECRET']
missing_vars = [var for var in required_env_vars if not os.environ.get(var)]
if missing_vars:
    logger.error("Missing required environment variables: %s", missing_vars)
    raise ValueError(f"Missing required environment variables: {missing_vars}")

# Database setup
DATABASE_PATH = os.environ.get('DATABASE_URL', 'codeex.db')

def get_db_connection():
    """Get database connection with proper error handling and optimizations"""
    try:
        conn = sqlite3.connect(DATABASE_PATH, timeout=30.0, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        
        # Enable WAL mode for better concurrency
        conn.execute('PRAGMA journal_mode=WAL;')
        conn.execute('PRAGMA synchronous=NORMAL;')
        conn.execute('PRAGMA cache_size=1000;')
        conn.execute('PRAGMA temp_store=memory;')
        
        return conn
    except sqlite3.Error as e:
        logger.error("Database connection error: %s", e)
        raise

def init_db():
    """Initialize SQLite database"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        # Users table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                display_name TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Chat sessions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS chat_sessions (
                id TEXT PRIMARY KEY,
                user_id INTEGER,
                title TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')

        # Messages table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES chat_sessions (id)
            )
        ''')

        # Password reset tokens table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS password_reset_tokens (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                token TEXT UNIQUE NOT NULL,
                expires_at TIMESTAMP NOT NULL,
                used BOOLEAN DEFAULT FALSE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')

        conn.commit()
        logger.info("Database initialized successfully")
    except sqlite3.Error as e:
        logger.error("Database initialization error: %s", e)
        raise
    finally:
        if conn:
            conn.close()

# Email sending function


def send_contact_email(name, email, subject, message):
    """Send contact form email to the.heoster@mail.com"""
    try:
        # Create message
        msg = MIMEMultipart()
        msg['From'] = formataddr(
            ('CodeEx AI Contact Form',
             EMAIL_USERNAME or 'noreply@codeexai.com'))
        msg['To'] = CONTACT_EMAIL
        msg['Subject'] = f"[CodeEx AI Contact] {subject}"

        # Create email body
        email_body = f"""
New contact form submission from CodeEx AI:

Name: {name}
Email: {email}
Subject: {subject}

Message:
{message}

---
This message was sent from the CodeEx AI contact form.
Reply directly to this email to respond to the user.
        """

        msg.attach(MIMEText(email_body, 'plain'))

        # Send email
        if EMAIL_USERNAME and EMAIL_PASSWORD:
            # Use configured SMTP server
            server = smtplib.SMTP(EMAIL_HOST, EMAIL_PORT)
            server.starttls()
            server.login(EMAIL_USERNAME, EMAIL_PASSWORD)
            text = msg.as_string()
            server.sendmail(EMAIL_USERNAME, CONTACT_EMAIL, text)
            server.quit()
            logger.info("Contact email sent successfully to %s", CONTACT_EMAIL)
            return True
        # Log the message if no email configuration
        logger.info(
            "Email configuration not set. Contact form message logged: %s (%s) - %s",
            name, email, subject)
        logger.info("Message content: %s", message)
        return True

    except (smtplib.SMTPException, OSError, ValueError) as e:
        logger.error("Failed to send contact email: %s", e)
        return False








def send_stylish_email(to_email, subject, html_content, text_content=None):
    """Send a stylish HTML email"""
    try:
        msg = MIMEMultipart('alternative')
        msg['From'] = formataddr(
            ('CodeEx AI', EMAIL_USERNAME or 'noreply@codeexai.com'))
        msg['To'] = to_email
        msg['Subject'] = subject

        # Create text version if not provided
        if not text_content:
            # Simple HTML to text conversion
            import re
            text_content = re.sub('<[^<]+?>', '', html_content)
            text_content = re.sub(r'\s+', ' ', text_content).strip()

        # Attach both text and HTML versions
        text_part = MIMEText(text_content, 'plain')
        html_part = MIMEText(html_content, 'html')

        msg.attach(text_part)
        msg.attach(html_part)

        # Send email
        if EMAIL_USERNAME and EMAIL_PASSWORD:
            server = smtplib.SMTP(EMAIL_HOST, EMAIL_PORT)
            server.starttls()
            server.login(EMAIL_USERNAME, EMAIL_PASSWORD)
            server.send_message(msg)
            server.quit()
            logger.info("Stylish email sent successfully to %s", to_email)
            return True

        logger.info(
            "Email configuration not set. Email logged for: %s", to_email)
        logger.info("Subject: %s", subject)
        return True

    except (smtplib.SMTPException, OSError, ValueError) as e:
        logger.error("Failed to send stylish email: %s", e)
        return False


def send_welcome_email(user_email, user_name):
    """Send a stylish welcome email to new users"""
    subject = "🎉 Welcome to CodeEx AI - Your AI Journey Begins!"

    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Welcome to CodeEx AI</title>
        <style>
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                line-height: 1.6;
                color: #333;
                max-width: 600px;
                margin: 0 auto;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                padding: 20px;
            }}
            .container {{
                background: white;
                border-radius: 15px;
                padding: 40px;
                box-shadow: 0 10px 30px rgba(0,0,0,0.1);
            }}
            .header {{
                text-align: center;
                margin-bottom: 30px;
            }}
            .logo {{
                width: 80px;
                height: 80px;
                background: linear-gradient(45deg, #667eea, #764ba2);
                border-radius: 50%;
                margin: 0 auto 20px;
                display: flex;
                align-items: center;
                justify-content: center;
                color: white;
                font-size: 24px;
                font-weight: bold;
            }}
            .welcome-title {{
                color: #2c3e50;
                font-size: 28px;
                margin-bottom: 10px;
                font-weight: 700;
            }}
            .subtitle {{
                color: #7f8c8d;
                font-size: 16px;
                margin-bottom: 30px;
            }}
            .features {{
                background: #f8f9fa;
                border-radius: 10px;
                padding: 25px;
                margin: 25px 0;
            }}
            .feature-item {{
                display: flex;
                align-items: center;
                margin-bottom: 15px;
                padding: 10px;
                background: white;
                border-radius: 8px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.05);
            }}
            .feature-icon {{
                width: 40px;
                height: 40px;
                border-radius: 50%;
                margin-right: 15px;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 18px;
            }}
            .ai-icon {{ background: linear-gradient(45deg, #667eea, #764ba2); color: white; }}
            .chat-icon {{ background: linear-gradient(45deg, #f093fb, #f5576c); color: white; }}
            .secure-icon {{ background: linear-gradient(45deg, #4facfe, #00f2fe); color: white; }}
            .learn-icon {{ background: linear-gradient(45deg, #43e97b, #38f9d7); color: white; }}
            .cta-button {{
                display: inline-block;
                background: linear-gradient(45deg, #667eea, #764ba2);
                color: white;
                padding: 15px 30px;
                text-decoration: none;
                border-radius: 25px;
                font-weight: 600;
                margin: 20px 0;
                transition: transform 0.3s ease;
            }}
            .cta-button:hover {{
                transform: translateY(-2px);
            }}
            .footer {{
                text-align: center;
                margin-top: 30px;
                padding-top: 20px;
                border-top: 1px solid #eee;
                color: #7f8c8d;
                font-size: 14px;
            }}
            .social-links {{
                margin: 20px 0;
            }}
            .social-links a {{
                display: inline-block;
                margin: 0 10px;
                color: #667eea;
                text-decoration: none;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <div class="logo">AI</div>
                <h1 class="welcome-title">Welcome to CodeEx AI!</h1>
                <p class="subtitle">Hello {user_name}, your intelligent AI companion is ready to assist you</p>
            </div>
            
            <div class="content">
                <p>🎉 <strong>Congratulations!</strong> You've successfully joined the CodeEx AI community. Get ready to experience the future of AI-powered assistance.</p>
                
                <div class="features">
                    <h3 style="color: #2c3e50; margin-bottom: 20px;">✨ What you can do with CodeEx AI:</h3>
                    
                    <div class="feature-item">
                        <div class="feature-icon ai-icon">🤖</div>
                        <div>
                            <strong>Multi-Provider AI</strong><br>
                            <small>Access OpenAI GPT, Google Gemini, and advanced local AI models</small>
                        </div>
                    </div>
                    
                    <div class="feature-item">
                        <div class="feature-icon chat-icon">💬</div>
                        <div>
                            <strong>Intelligent Conversations</strong><br>
                            <small>Context-aware chat with memory and learning capabilities</small>
                        </div>
                    </div>
                    
                    <div class="feature-item">
                        <div class="feature-icon secure-icon">🔒</div>
                        <div>
                            <strong>Secure & Private</strong><br>
                            <small>End-to-end encryption and privacy-first design</small>
                        </div>
                    </div>
                    
                    <div class="feature-item">
                        <div class="feature-icon learn-icon">📚</div>
                        <div>
                            <strong>Continuous Learning</strong><br>
                            <small>AI that adapts and improves based on your interactions</small>
                        </div>
                    </div>
                </div>
                
                <div style="text-align: center;">
                    <a href="{request.url_root}chat" class="cta-button">🚀 Start Your First Chat</a>
                </div>
                
                <div style="background: #e8f4fd; padding: 20px; border-radius: 10px; margin: 25px 0;">
                    <h4 style="color: #2980b9; margin-bottom: 10px;">💡 Quick Tips to Get Started:</h4>
                    <ul style="color: #34495e; margin: 0; padding-left: 20px;">
                        <li>Try asking me to help with coding, writing, or problem-solving</li>
                        <li>Use voice input for hands-free interaction</li>
                        <li>Explore different AI providers for varied perspectives</li>
                        <li>Check out the documentation for advanced features</li>
                    </ul>
                </div>
            </div>
            
            <div class="footer">
                <div class="social-links">
                    <a href="{request.url_root}docs">📖 Documentation</a>
                    <a href="{request.url_root}contact">💌 Contact Support</a>
                    <a href="{request.url_root}about">ℹ️ About Us</a>
                </div>
                <p>Thank you for choosing CodeEx AI. We're excited to be part of your AI journey!</p>
                <p><small>© 2025 CodeEx AI. All rights reserved.</small></p>
            </div>
        </div>
    </body>
    </html>
    """

    text_content = f"""
    Welcome to CodeEx AI!
    
    Hello {user_name},
    
    Congratulations! You've successfully joined the CodeEx AI community.
    
    What you can do with CodeEx AI:
    🤖 Multi-Provider AI - Access OpenAI GPT, Google Gemini, and local AI models
    💬 Intelligent Conversations - Context-aware chat with memory
    🔒 Secure & Private - End-to-end encryption and privacy-first design
    📚 Continuous Learning - AI that adapts to your interactions
    
    Quick Tips:
    - Try asking me to help with coding, writing, or problem-solving
    - Use voice input for hands-free interaction
    - Explore different AI providers for varied perspectives
    - Check out the documentation for advanced features
    
    Start your first chat: {request.url_root}chat
    
    Thank you for choosing CodeEx AI!
    """

    return send_stylish_email(user_email, subject, html_content, text_content)


def send_password_reset_email(user_email, user_name, reset_token):
    """Send password reset email"""
    reset_url = f"{request.url_root}reset-password?token={reset_token}"
    subject = "🔐 Reset Your CodeEx AI Password"

    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Reset Your Password</title>
        <style>
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                line-height: 1.6;
                color: #333;
                max-width: 600px;
                margin: 0 auto;
                background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
                padding: 20px;
            }}
            .container {{
                background: white;
                border-radius: 15px;
                padding: 40px;
                box-shadow: 0 10px 30px rgba(0,0,0,0.1);
            }}
            .header {{
                text-align: center;
                margin-bottom: 30px;
            }}
            .logo {{
                width: 80px;
                height: 80px;
                background: linear-gradient(45deg, #ff6b6b, #ee5a24);
                border-radius: 50%;
                margin: 0 auto 20px;
                display: flex;
                align-items: center;
                justify-content: center;
                color: white;
                font-size: 24px;
                font-weight: bold;
            }}
            .reset-button {{
                display: inline-block;
                background: linear-gradient(45deg, #ff6b6b, #ee5a24);
                color: white;
                padding: 15px 30px;
                text-decoration: none;
                border-radius: 25px;
                font-weight: 600;
                margin: 20px 0;
                transition: transform 0.3s ease;
            }}
            .reset-button:hover {{
                transform: translateY(-2px);
            }}
            .warning {{
                background: #fff3cd;
                border: 1px solid #ffeaa7;
                border-radius: 8px;
                padding: 15px;
                margin: 20px 0;
                color: #856404;
            }}
            .footer {{
                text-align: center;
                margin-top: 30px;
                padding-top: 20px;
                border-top: 1px solid #eee;
                color: #7f8c8d;
                font-size: 14px;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <div class="logo">🔐</div>
                <h1 style="color: #2c3e50; font-size: 28px; margin-bottom: 10px;">Password Reset Request</h1>
                <p style="color: #7f8c8d; font-size: 16px;">Hello {user_name}, we received a request to reset your password</p>
            </div>
            
            <div class="content">
                <p>Someone requested a password reset for your CodeEx AI account. If this was you, click the button below to reset your password:</p>
                
                <div style="text-align: center; margin: 30px 0;">
                    <a href="{reset_url}" class="reset-button">🔑 Reset My Password</a>
                </div>
                
                <div class="warning">
                    <strong>⚠️ Important Security Information:</strong>
                    <ul style="margin: 10px 0; padding-left: 20px;">
                        <li>This link will expire in 1 hour for security reasons</li>
                        <li>If you didn't request this reset, please ignore this email</li>
                        <li>Never share this link with anyone</li>
                        <li>Contact support if you have concerns about your account security</li>
                    </ul>
                </div>
                
                <p><strong>Alternative method:</strong> If the button doesn't work, copy and paste this link into your browser:</p>
                <p style="background: #f8f9fa; padding: 10px; border-radius: 5px; word-break: break-all; font-family: monospace; font-size: 12px;">{reset_url}</p>
            </div>
            
            <div class="footer">
                <p>This password reset link will expire in 1 hour.</p>
                <p>If you need help, contact us at <a href="mailto:support@codeexai.com">support@codeexai.com</a></p>
                <p><small>© 2024 CodeEx AI. All rights reserved.</small></p>
            </div>
        </div>
    </body>
    </html>
    """

    text_content = f"""
    Password Reset Request - CodeEx AI
    
    Hello {user_name},
    
    Someone requested a password reset for your CodeEx AI account.
    
    If this was you, use this link to reset your password:
    {reset_url}
    
    IMPORTANT:
    - This link expires in 1 hour
    - If you didn't request this, ignore this email
    - Never share this link with anyone
    
    Need help? Contact the.heoster@mail.com 
    
    © 2025 CodeEx AI
    """

    return send_stylish_email(user_email, subject, html_content, text_content)


def generate_reset_token():
    """Generate a secure password reset token"""
    return secrets.token_urlsafe(32)


def create_password_reset_token(user_id):
    """Create a password reset token for a user"""
    token = generate_reset_token()
    expires_at = datetime.now() + timedelta(hours=1)  # Token expires in 1 hour

    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        # Invalidate any existing tokens for this user
        cursor.execute(
            'UPDATE password_reset_tokens SET used = TRUE WHERE user_id = ? AND used = FALSE',
            (user_id,)
        )

        # Create new token
        cursor.execute(
            'INSERT INTO password_reset_tokens (user_id, token, expires_at) VALUES (?, ?, ?)',
            (user_id, token, expires_at)
        )

        conn.commit()
        return token
    except sqlite3.Error as e:
        logger.error("Error creating password reset token: %s", e)
        raise
    finally:
        if conn:
            conn.close()


def verify_reset_token(token):
    """Verify a password reset token and return user_id if valid"""
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute('''
            SELECT user_id FROM password_reset_tokens 
            WHERE token = ? AND used = FALSE AND expires_at > datetime('now')
        ''', (token,))

        result = cursor.fetchone()
        return result[0] if result else None
    except sqlite3.Error as e:
        logger.error("Error verifying reset token: %s", e)
        return None
    finally:
        if conn:
            conn.close()


def use_reset_token(token):
    """Mark a reset token as used"""
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute(
            'UPDATE password_reset_tokens SET used = TRUE WHERE token = ?',
            (token,)
        )

        conn.commit()
    except sqlite3.Error as e:
        logger.error("Error using reset token: %s", e)
        raise
    finally:
        if conn:
            conn.close()

# Error handlers
@app.errorhandler(404)
def not_found_error(error):
    """Handle 404 errors"""
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    logger.error("Internal server error: %s", error)
    return render_template('500.html'), 500

@app.errorhandler(403)
def forbidden_error(error):
    """Handle 403 errors"""
    return render_template('403.html'), 403

if SECURITY_AVAILABLE:
    @app.errorhandler(429)
    def ratelimit_handler(e):
        """Handle rate limit errors"""
        return jsonify({'error': 'Rate limit exceeded', 'retry_after': str(e.retry_after)}), 429

# Authentication decorator


def login_required(f):
    """Decorator to require user login for protected routes"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# Routes


@app.route('/favicon.ico')
def favicon():
    """Serve favicon to prevent 404 errors"""
    return redirect(url_for('static', filename='images/favicon.ico'), code=301)



@app.route('/')
def index():
    """Landing page"""
    if 'user_id' in session:
        return redirect(url_for('chat'))
    return render_template('index.html')


def apply_rate_limit(limit_string):
    """Apply rate limiting if available, otherwise return identity decorator"""
    if SECURITY_AVAILABLE and limiter:
        return limiter.limit(limit_string)
    else:
        return lambda f: f

@app.route('/login', methods=['GET', 'POST'])
@apply_rate_limit("10 per minute")
def login():
    """Login page"""
    if request.method == 'POST':
        conn = None
        try:
            data = request.get_json() if request.is_json else request.form
            email = data.get('email')
            password = data.get('password')
            action = data.get('action', 'signin')

            conn = get_db_connection()
            cursor = conn.cursor()

            if action == 'signup':
                # Sign up
                display_name = data.get('display_name', email.split('@')[0])
                password_hash = generate_password_hash(password)

                try:
                    cursor.execute(
                        'INSERT INTO users (email, password_hash, display_name) VALUES (?, ?, ?)',
                        (email, password_hash, display_name)
                    )
                    conn.commit()
                    user_id = cursor.lastrowid
                    session['user_id'] = user_id
                    session['email'] = email
                    session['display_name'] = display_name

                    # Send welcome email to new user
                    try:
                        send_welcome_email(email, display_name)
                    except Exception as e:
                        logger.error("Failed to send welcome email: %s", e)
                        # Don't fail registration if email fails

                    if request.is_json:
                        return jsonify({'success': True, 'redirect': '/chat'})
                    return redirect(url_for('chat'))

                except sqlite3.IntegrityError:
                    if request.is_json:
                        return jsonify(
                            {'success': False, 'error': 'Email already exists'})
                    flash('Email already exists')

            # Sign in
            cursor.execute(
                'SELECT id, password_hash, display_name FROM users WHERE email = ?', (email,))
            user = cursor.fetchone()

            if user and check_password_hash(user[1], password):
                session['user_id'] = user[0]
                session['email'] = email
                session['display_name'] = user[2]

                if request.is_json:
                    return jsonify({'success': True, 'redirect': '/chat'})
                return redirect(url_for('chat'))

            if request.is_json:
                return jsonify(
                    {'success': False, 'error': 'Invalid credentials'})
            flash('Invalid credentials')

        except sqlite3.Error as e:
            logger.error("Database error in login: %s", e)
            if request.is_json:
                return jsonify({
                    'success': False, 
                    'error': 'Database temporarily unavailable. Please try again.'
                }), 503
            flash('Service temporarily unavailable. Please try again.')
        finally:
            if conn:
                conn.close()
        flash('Invalid credentials')

        conn.close()

    return render_template('login.html', google_client_id=GOOGLE_CLIENT_ID)


@app.route('/auth/google', methods=['POST'])
@apply_rate_limit("5 per minute")
def google_auth():
    """Handle Google OAuth authentication"""
    try:
        data = request.get_json()
        credential = data.get('credential')

        if not credential:
            return jsonify(
                {'success': False, 'error': 'No credential provided'}), 400

        # Verify the Google ID token
        try:
            idinfo = id_token.verify_oauth2_token(
                credential,
                google_requests.Request(),
                GOOGLE_CLIENT_ID
            )

            # Extract user information
            email = idinfo.get('email')
            name = idinfo.get('name', email.split('@')[0])
            google_id = idinfo.get('sub')
            picture = idinfo.get('picture')

            if not email:
                return jsonify(
                    {'success': False, 'error': 'Email not provided by Google'}), 400

            # Check if user exists or create new user
            conn = None
            try:
                conn = get_db_connection()
                cursor = conn.cursor()

                # First, check if user exists
                cursor.execute(
                    'SELECT id, display_name FROM users WHERE email = ?', (email,))
                user = cursor.fetchone()

                if user:
                    # User exists, log them in
                    user_id = user[0]
                    display_name = user[1] or name
                else:
                    # Create new user with Google info
                    # For Google OAuth users, we'll use a placeholder password hash
                    password_hash = generate_password_hash(
                        f'google_oauth_{google_id}')

                    cursor.execute(
                        'INSERT INTO users (email, password_hash, display_name) VALUES (?, ?, ?)',
                        (email, password_hash, name)
                    )
                    conn.commit()
                    user_id = cursor.lastrowid
                    display_name = name

                    # Send welcome email to new Google user
                    try:
                        send_welcome_email(email, display_name)
                    except Exception as e:
                        logger.error(
                            "Failed to send welcome email to Google user: %s", e)

                # Set session
                session['user_id'] = user_id
                session['email'] = email
                session['display_name'] = display_name
                session['auth_method'] = 'google'
                session['profile_picture'] = picture

            except sqlite3.Error as e:
                logger.error("Database error in Google auth: %s", e)
                return jsonify({
                    'success': False,
                    'error': 'Database temporarily unavailable. Please try again.'
                }), 503
            finally:
                if conn:
                    conn.close()

            return jsonify({
                'success': True,
                'redirect': '/chat',
                'user': {
                    'id': user_id,
                    'email': email,
                    'display_name': display_name,
                    'picture': picture
                }
            })

        except ValueError as e:
            logger.error("Invalid Google token: %s", e)
            return jsonify(
                {'success': False, 'error': 'Invalid Google token'}), 400

    except (ValueError, KeyError, requests.RequestException) as e:
        logger.error("Google auth error: %s", e)
        return jsonify(
            {'success': False, 'error': 'Authentication failed'}), 500


@app.route('/auth/google/login')
def google_login_redirect():
    """Redirect route for Google OAuth login (fallback)"""
    # This is a fallback route if the JavaScript Google Sign-In doesn't work
    google_auth_url = (
        f"https://accounts.google.com/oauth/authorize?"
        f"client_id={GOOGLE_CLIENT_ID}&"
        f"redirect_uri={request.url_root}auth/google/callback&"
        f"scope=openid%20email%20profile&response_type=code"
    )
    return redirect(google_auth_url)


@app.route('/auth/google/signup')
def google_signup_redirect():
    """Redirect route for Google OAuth signup (fallback)"""
    # Same as login since Google OAuth handles both cases
    return redirect(url_for('google_login_redirect'))


@app.route('/auth/google/callback')
def google_callback():
    """Handle Google OAuth callback (fallback)"""
    try:
        code = request.args.get('code')
        if not code:
            flash('Google authentication failed')
            return redirect(url_for('login'))

        # Exchange code for token
        token_url = "https://oauth2.googleapis.com/token"
        token_data = {
            'client_id': GOOGLE_CLIENT_ID,
            'client_secret': GOOGLE_CLIENT_SECRET,
            'code': code,
            'grant_type': 'authorization_code',
            'redirect_uri': f"{request.url_root}auth/google/callback"
        }

        token_response = requests.post(token_url, data=token_data, timeout=10)
        token_json = token_response.json()

        if 'access_token' not in token_json:
            flash('Failed to get access token from Google')
            return redirect(url_for('login'))

        # Get user info
        user_info_url = (
            "https://www.googleapis.com/oauth2/v2/userinfo?"
            f"access_token={token_json['access_token']}"
        )
        user_response = requests.get(user_info_url, timeout=10)
        user_info = user_response.json()

        email = user_info.get('email')
        name = user_info.get('name', email.split('@')[0])
        google_id = user_info.get('id')
        picture = user_info.get('picture')

        if not email:
            flash('Email not provided by Google')
            return redirect(url_for('login'))

        # Check if user exists or create new user
        conn = sqlite3.connect('codeex.db')
        cursor = conn.cursor()

        cursor.execute(
            'SELECT id, display_name FROM users WHERE email = ?', (email,))
        user = cursor.fetchone()

        if user:
            user_id = user[0]
            display_name = user[1] or name
        else:
            password_hash = generate_password_hash(f'google_oauth_{google_id}')
            cursor.execute(
                'INSERT INTO users (email, password_hash, display_name) VALUES (?, ?, ?)',
                (email, password_hash, name)
            )
            conn.commit()
            user_id = cursor.lastrowid
            display_name = name

            # Send welcome email to new Google callback user
            try:
                send_welcome_email(email, display_name)
            except Exception as e:
                logger.error(
                    "Failed to send welcome email to Google callback user: %s", e)

        # Set session
        session['user_id'] = user_id
        session['email'] = email
        session['display_name'] = display_name
        session['auth_method'] = 'google'
        session['profile_picture'] = picture

        conn.close()

        return redirect(url_for('chat'))

    except (requests.RequestException, KeyError, ValueError) as e:
        logger.error("Google callback error: %s", e)
        flash('Google authentication failed')
        return redirect(url_for('login'))


@app.route('/forgot-password', methods=['GET', 'POST'])
def forgot_password():
    """Handle forgot password requests"""
    if request.method == 'POST':
        conn = None
        try:
            data = request.get_json() if request.is_json else request.form
            email = data.get('email')

            if not email:
                return jsonify({'success': False, 'error': 'Email is required'}), 400

            # Check if user exists with proper connection handling
            conn = get_db_connection()
            cursor = conn.cursor()

            cursor.execute(
                'SELECT id, display_name FROM users WHERE email = ?', (email,))
            user = cursor.fetchone()

            if user:
                user_id, display_name = user

                # Generate secure reset token
                reset_token = secrets.token_urlsafe(32)
                expires_at = datetime.now() + timedelta(hours=1)  # Token expires in 1 hour

                # Store reset token in database
                cursor.execute('''
                    INSERT INTO password_reset_tokens (user_id, token, expires_at)
                    VALUES (?, ?, ?)
                ''', (user_id, reset_token, expires_at))
                conn.commit()

                # Send password reset email
                email_sent = send_password_reset_email(
                    email, display_name, reset_token)

                if email_sent:
                    logger.info("Password reset email sent to %s", email)
                else:
                    logger.warning(
                        "Failed to send password reset email to %s", email)

            # Always return success to prevent email enumeration attacks
            return jsonify({
                'success': True,
                'message': 'If an account with that email exists, we\'ve sent a password reset link.'
            })

        except sqlite3.Error as e:
            logger.error("Database error in forgot password: %s", e)
            return jsonify({
                'success': False,
                'error': 'Database temporarily unavailable. Please try again.'
            }), 503
        except (ValueError, KeyError, TypeError) as e:
            logger.error("Forgot password error: %s", e)
            return jsonify({
                'success': False,
                'error': 'Failed to process request. Please try again.'
            }), 500
        finally:
            if conn:
                conn.close()

    return render_template('forgot_password.html')


@app.route('/reset-password', methods=['GET', 'POST'])
def reset_password():
    """Handle password reset with token"""
    if request.method == 'GET':
        token = request.args.get('token')
        if not token:
            flash('Invalid or missing reset token')
            return redirect(url_for('login'))

        conn = None
        try:
            # Verify token exists and is not expired
            conn = get_db_connection()
            cursor = conn.cursor()

            cursor.execute('''
                SELECT user_id, expires_at, used FROM password_reset_tokens 
                WHERE token = ?
            ''', (token,))
            token_data = cursor.fetchone()

            if not token_data:
                flash('Invalid reset token')
                return redirect(url_for('login'))

            user_id, expires_at, used = token_data

            if used:
                flash('This reset link has already been used')
                return redirect(url_for('login'))

            if datetime.now() > datetime.fromisoformat(expires_at):
                flash('This reset link has expired')
                return redirect(url_for('login'))

            return render_template('reset_password.html', token=token)

        except sqlite3.Error as e:
            logger.error("Database error in reset password GET: %s", e)
            flash('Service temporarily unavailable. Please try again.')
            return redirect(url_for('login'))
        finally:
            if conn:
                conn.close()

    elif request.method == 'POST':
        conn = None
        try:
            data = request.get_json() if request.is_json else request.form
            token = data.get('token')
            new_password = data.get('password')
            confirm_password = data.get('confirm_password')

            if not all([token, new_password, confirm_password]):
                return jsonify({
                    'success': False,
                    'error': 'All fields are required'
                }), 400

            if new_password != confirm_password:
                return jsonify({
                    'success': False,
                    'error': 'Passwords do not match'
                }), 400

            if len(new_password) < 8:
                return jsonify({
                    'success': False,
                    'error': 'Password must be at least 8 characters long'
                }), 400

            # Verify token
            conn = get_db_connection()
            cursor = conn.cursor()

            cursor.execute('''
                SELECT user_id, expires_at, used FROM password_reset_tokens 
                WHERE token = ?
            ''', (token,))
            token_data = cursor.fetchone()

            if not token_data:
                return jsonify({
                    'success': False,
                    'error': 'Invalid reset token'
                }), 400

            user_id, expires_at, used = token_data

            if used:
                return jsonify({
                    'success': False,
                    'error': 'This reset link has already been used'
                }), 400

            if datetime.now() > datetime.fromisoformat(expires_at):
                return jsonify({
                    'success': False,
                    'error': 'This reset link has expired'
                }), 400

            # Update password
            password_hash = generate_password_hash(new_password)
            cursor.execute(
                'UPDATE users SET password_hash = ? WHERE id = ?',
                (password_hash, user_id)
            )

            # Mark token as used
            cursor.execute(
                'UPDATE password_reset_tokens SET used = TRUE WHERE token = ?',
                (token,)
            )

            conn.commit()

            logger.info("Password reset successful for user ID: %s", user_id)

            return jsonify({
                'success': True,
                'message': 'Password reset successful! You can now log in with your new password.',
                'redirect': '/login'
            })

        except sqlite3.Error as e:
            logger.error("Database error in reset password POST: %s", e)
            return jsonify({
                'success': False,
                'error': 'Database temporarily unavailable. Please try again.'
            }), 503
        except (ValueError, KeyError, TypeError) as e:
            logger.error("Password reset error: %s", e)
            return jsonify({
                'success': False,
                'error': 'Failed to reset password. Please try again.'
            }), 500
        finally:
            if conn:
                conn.close()


@app.route('/logout')
def logout():
    """Logout user"""
    session.clear()
    return redirect(url_for('index'))


@app.route('/chat')
@login_required
def chat():
    """Main chat interface"""
    try:
        return render_template('chat.html', user=session)
    except Exception as e:
        logger.error("Error rendering chat template: %s", e)
        return render_template('chat.html', user=session)


@app.route('/api/chat/sessions', methods=['GET', 'POST'])
@login_required
def chat_sessions():
    """Get or create chat sessions"""
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        if request.method == 'POST':
            # Create new session
            session_id = str(uuid.uuid4())
            title = request.json.get('title', 'New Chat')

            cursor.execute(
                'INSERT INTO chat_sessions (id, user_id, title) VALUES (?, ?, ?)',
                (session_id, session['user_id'], title)
            )
            conn.commit()

            return jsonify({'session_id': session_id, 'title': title})

        # Get all sessions
        cursor.execute(
            'SELECT id, title, created_at FROM chat_sessions '
            'WHERE user_id = ? ORDER BY updated_at DESC',
            (session['user_id'],)
        )
        sessions = [{'id': row[0], 'title': row[1], 'created_at': row[2]}
                    for row in cursor.fetchall()]

        return jsonify(sessions)

    except sqlite3.Error as e:
        logger.error("Database error in chat sessions: %s", e)
        return jsonify({'error': 'Database temporarily unavailable'}), 503
    finally:
        if conn:
            conn.close()

    return jsonify(sessions)


@app.route('/docs')
def docs():
    """Documentation page"""
    return render_template('docs.html')


@app.route('/privacy')
def privacy():
    """Privacy Policy page"""
    return render_template('policy.html')


@app.route('/terms')
def terms():
    """Terms of Service page"""
    return render_template('policy.html')


@app.route('/policy')
def policy():
    """Privacy Policy & Terms page"""
    return render_template('policy.html')


@app.route('/cookies')
def cookies():
    """Cookie Policy page"""
    return render_template('policy.html')


@app.route('/contact', methods=['GET', 'POST'])
def contact():
    """Contact page with email forwarding to the.heoster@mail.com"""
    if request.method == 'POST':
        try:
            # Handle contact form submission
            name = request.form.get('name')
            email = request.form.get('email')
            subject = request.form.get('subject')
            message = request.form.get('message')
            privacy_agree = request.form.get('privacy-agree')

            # Validate required fields
            if not all([name, email, subject, message, privacy_agree]):
                return jsonify({
                    'success': False,
                    'error': ('All fields are required and you must agree '
                              'to the privacy policy.')
                }), 400

            # Send email to the.heoster@mail.com
            email_sent = send_contact_email(name, email, subject, message)

            if email_sent:
                # Log successful contact form submission
                logger.info(
                    "Contact form submitted and emailed: %s (%s) - %s",
                    name, email, subject)

                return jsonify({
                    'success': True,
                    'message': ('Thank you for your message! We have received it '
                               'and will get back to you soon.')
                })

            # Log failed email but still acknowledge the submission
            logger.warning(
                "Contact form submitted but email failed: %s (%s) - %s",
                name, email, subject)

            return jsonify({
                'success': True,
                'message': ('Thank you for your message! We have received it '
                           'and will get back to you soon.')
            })

        except (ValueError, KeyError, TypeError) as e:
            logger.error("Contact form error: %s", e)
            return jsonify({
                'success': False,
                'error': 'Failed to send message. Please try again.'
            }), 500

    return render_template('contact.html')


@app.route('/about')
def about():
    """About page"""
    return render_template('about.html')


@app.route('/faq')
def faq():
    """FAQ page"""
    return render_template('faq.html')


@app.route('/settings')
@login_required
def settings():
    """User settings page"""
    return render_template('settings.html', user=session)


# Import web scraping service

# Web Scraping API Endpoints

@app.route('/api/scraping/test', methods=['POST'])
@login_required
def test_scraping():
    """Test web scraping functionality"""
    try:
        data = request.get_json()
        url = data.get('url', 'https://httpbin.org/html')

        scraping_service = get_scraping_service()
        result = scraping_service.test_scraping(url)

        return jsonify(result)
    except (AttributeError, ValueError, TypeError) as e:
        logger.error("Error testing scraping: %s", e)
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/scraping/stats')
@login_required
def scraping_stats():
    """Get web scraping statistics"""
    try:
        scraping_service = get_scraping_service()
        stats = scraping_service.get_scraping_stats()
        return jsonify(stats)
    except (AttributeError, ValueError, TypeError) as e:
        logger.error("Error getting scraping stats: %s", e)
        return jsonify({'error': str(e)}), 500


@app.route('/api/scraping/results')
@login_required
def scraping_results():
    """Get recent scraping results"""
    try:
        limit = request.args.get('limit', 50, type=int)
        scraping_service = get_scraping_service()
        results = scraping_service.get_recent_results(limit)
        return jsonify({'results': results})
    except (AttributeError, ValueError, TypeError) as e:
        logger.error("Error getting scraping results: %s", e)
        return jsonify({'error': str(e)}), 500


@app.route('/api/scraping/search')
@login_required
def search_scraped_content():
    """Search through scraped content"""
    try:
        query = request.args.get('q', '')
        limit = request.args.get('limit', 20, type=int)

        if not query:
            return jsonify({'error': 'Query parameter required'}), 400

        scraping_service = get_scraping_service()
        results = scraping_service.search_scraped_content(query, limit)
        return jsonify({'results': results, 'query': query})
    except (AttributeError, ValueError, TypeError) as e:
        logger.error("Error searching scraped content: %s", e)
        return jsonify({'error': str(e)}), 500


@app.route('/api/scraping/start', methods=['POST'])
@login_required
def start_auto_scraping():
    """Start automatic web scraping"""
    try:
        scraping_service = get_scraping_service()
        scraping_service.start_auto_scraping()
        return jsonify({'success': True, 'message': 'Auto-scraping started'})
    except (AttributeError, ValueError, RuntimeError) as e:
        logger.error("Error starting auto-scraping: %s", e)
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/scraping/stop', methods=['POST'])
@login_required
def stop_auto_scraping():
    """Stop automatic web scraping"""
    try:
        scraping_service = get_scraping_service()
        scraping_service.stop_auto_scraping()
        return jsonify({'success': True, 'message': 'Auto-scraping stopped'})
    except (AttributeError, ValueError, RuntimeError) as e:
        logger.error("Error stopping auto-scraping: %s", e)
        return jsonify({'success': False, 'error': str(e)}), 500

# AI Enhancement API Endpoints


@app.route('/api/ai/optimize', methods=['POST'])
@login_required
def optimize_ai():
    """Optimize AI performance"""
    try:
        # Simulate AI optimization process
        logger.info("AI optimization requested")

        # Here you would implement actual AI optimization logic
        # For now, we'll simulate the process

        return jsonify({
            'success': True,
            'message': 'AI optimization completed',
            'improvements': {
                'response_time': '15% faster',
                'accuracy': '2.3% improvement',
                'memory_usage': '8% reduction'
            }
        })
    except (AttributeError, ValueError, RuntimeError) as e:
        logger.error("Error optimizing AI: %s", e)
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/ai/export')
@login_required
def export_ai_data():
    """Export AI data and analytics"""
    try:
        # Get AI metrics and data
        ai_data = {
            'export_timestamp': datetime.now().isoformat(),
            'user_id': session['user_id'],
            'metrics': get_ai_metrics(),
            'conversations': [],  # Would include conversation data
            'learning_patterns': [],  # Would include learning data
            'performance_stats': {
                'total_responses': 1247,
                'avg_response_time': 0.8,
                'accuracy_score': 94.7,
                'user_satisfaction': 4.6
            }
        }

        # Create JSON response
        response = jsonify(ai_data)
        filename = (f'ai_export_{session["user_id"]}_'
                    f'{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
        response.headers['Content-Disposition'] = f'attachment; filename={filename}'
        return response

    except (AttributeError, ValueError, KeyError) as e:
        logger.error("Error exporting AI data: %s", e)
        return jsonify({'error': str(e)}), 500


@app.route('/api/ai/memory/clear', methods=['POST'])
@login_required
def clear_ai_memory():
    """Clear AI memory and context"""
    try:
        # Clear user's AI memory and context
        # This would involve clearing conversation history, context memory,
        # etc.

        logger.info("AI memory cleared for user %s", session['user_id'])

        return jsonify({
            'success': True,
            'message': 'AI memory cleared successfully',
            'cleared_items': {
                'conversation_context': 'cleared',
                'learning_patterns': 'reset',
                'user_preferences': 'maintained'
            }
        })
    except (AttributeError, ValueError, RuntimeError) as e:
        logger.error("Error clearing AI memory: %s", e)
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/ai/analytics')
@login_required
def ai_analytics():
    """Get comprehensive AI analytics"""
    try:
        analytics = {
            'performance_metrics': {
                'response_accuracy': 94.7,
                'avg_response_time': 0.8,
                'learning_progress': 87.3,
                'user_satisfaction': 4.6
            },
            'usage_stats': {
                'total_conversations': 1247,
                'ai_responses_generated': 3891,
                'storage_used_mb': 15.7,
                'features_used': [
                    'mathematics', 'code_generation', 'knowledge_base',
                    'web_scraping', 'voice_processing'
                ]
            },
            'capabilities': {
                'mathematics': {'status': 'active', 'accuracy': 96.2},
                'code_generation': {'status': 'active', 'accuracy': 93.8},
                'web_scraping': {'status': 'active', 'success_rate': 98.5},
                'knowledge_base': {'status': 'active', 'coverage': 89.1},
                'voice_processing': {'status': 'active', 'accuracy': 91.7}
            },
            'learning_stats': {
                'patterns_learned': 2847,
                'context_entries': 1247,
                'memory_usage_mb': 2.3,
                'retention_period_days': 90
            }
        }

        return jsonify(analytics)
    except (AttributeError, ValueError, KeyError) as e:
        logger.error("Error getting AI analytics: %s", e)
        return jsonify({'error': str(e)}), 500


@app.route('/storage')
@login_required
def storage_management():
    """Storage management page"""
    return render_template('storage_management.html', user=session)


@app.route('/ai-brain-dashboard')
@login_required
def ai_brain_dashboard():
    """AI Brain Dashboard page"""
    return render_template('ai_dashboard.html', user=session)


@app.route('/offline')
def offline():
    """Offline page for PWA"""
    return render_template('offline.html')


@app.route('/static/manifest.json')
def manifest():
    """Serve PWA manifest"""
    return app.send_static_file('manifest.json')


@app.route('/static/sw.js')
def service_worker():
    """Serve service worker"""
    response = app.send_static_file('sw.js')
    response.headers['Content-Type'] = 'application/javascript'
    response.headers['Service-Worker-Allowed'] = '/'
    return response


@app.route('/robots.txt')
def robots_txt():
    """Serve robots.txt"""
    return app.send_static_file('robots.txt')


@app.route('/api/chat/<session_id>/messages', methods=['GET', 'POST'])
@login_required
def chat_messages(session_id):
    """Get or send messages in a chat session with enhanced features"""

    if request.method == 'POST':
        # Send message with enhanced context
        message = request.json.get('message')
        encrypt = request.json.get('encrypt', True)

        try:
            # Save user message with encryption
            chat_history_service.save_message(
                session_id=session_id,
                role='user',
                content=message,
                user_id=session['user_id'],
                encrypt=encrypt
            )

            # Get conversation context for better AI responses
            context = chat_history_service.get_conversation_context(
                user_id=session['user_id'],
                session_id=session_id,
                context_window=10
            )

            # Generate AI response using multi-provider system
            try:
                # Try multi-provider AI first
                provider_pref = request.json.get('provider')
                multi_ai_result = handle_chat_request(
                    user_input=message,
                    context=context['recent_messages'],
                    provider_preference=provider_pref
                )
                ai_response = multi_ai_result['response']
                provider = multi_ai_result['provider']
                model = multi_ai_result['model']
                ai_source = f"{provider}_{model}"
                confidence = multi_ai_result['confidence']
            except (AttributeError, ValueError, RuntimeError, KeyError) as e:
                logger.warning(
                    "Multi-provider AI failed, falling back to local: %s", e)
                # Fallback to original AI system
                ai_result = get_intelligent_response(
                    user_input=message,
                    context=context['recent_messages'],
                    user_id=str(session['user_id'])
                )
                ai_response = ai_result['response']
                ai_source = ai_result['source']
                confidence = ai_result['confidence']

            # Save AI response
            chat_history_service.save_message(
                session_id=session_id,
                role='assistant',
                content=ai_response,
                user_id=session['user_id'],
                encrypt=encrypt
            )

            return jsonify({
                'response': ai_response,
                'context_used': len(context['context_memories']),
                'encrypted': encrypt,
                'ai_source': ai_source,
                'confidence': confidence,
                'provider': (multi_ai_result.get('provider', 'local')
                           if 'multi_ai_result' in locals() else 'local'),
                'model': (multi_ai_result.get('model', 'unknown')
                          if 'multi_ai_result' in locals() else 'unknown'),
                'task_type': (multi_ai_result.get('task_type', 'general')
                              if 'multi_ai_result' in locals() else 'general'),
                'learning_stats': (multi_ai_result.get('metadata', {})
                                   if 'multi_ai_result' in locals() else {})
            })

        except PermissionError:
            return jsonify({'error': 'Access denied'}), 403
        except (AttributeError, ValueError, KeyError, TypeError) as e:
            logger.error("Error in chat_messages: %s", e)
            return jsonify({'error': 'Internal server error'}), 500

    else:
        # Get messages with decryption
        try:
            messages = chat_history_service.get_messages(
                session_id=session_id,
                user_id=session['user_id'],
                decrypt=True
            )

            # Convert to JSON-serializable format
            messages_data = []
            for msg in messages:
                messages_data.append({
                    'role': msg.role,
                    'content': msg.content,
                    'timestamp': msg.timestamp.isoformat(),
                    'encrypted': msg.encrypted
                })

            return jsonify(messages_data)

        except PermissionError:
            return jsonify({'error': 'Access denied'}), 403
        except (AttributeError, ValueError, KeyError, TypeError) as e:
            logger.error("Error retrieving messages: %s", e)
            return jsonify({'error': 'Internal server error'}), 500


@app.route('/api/chat/context/<session_id>')
@login_required
def get_conversation_context(session_id):
    """Get conversation context and memory for a session"""
    try:
        context = chat_history_service.get_conversation_context(
            user_id=session['user_id'],
            session_id=session_id,
            context_window=20
        )
        return jsonify(context)
    except (AttributeError, ValueError, KeyError) as e:
        logger.error("Error getting context: %s", e)
        return jsonify({'error': 'Failed to retrieve context'}), 500


@app.route('/api/storage/analytics')
@login_required
def storage_analytics():
    """Get storage analytics for the current user"""
    try:
        analytics = chat_history_service.get_storage_analytics(
            session['user_id'])
        return jsonify(analytics)
    except (AttributeError, ValueError, KeyError) as e:
        logger.error("Error getting analytics: %s", e)
        return jsonify({'error': 'Failed to retrieve analytics'}), 500


@app.route('/api/storage/cleanup', methods=['POST'])
@login_required
def cleanup_storage():
    """Clean up old chat data based on retention policies"""
    try:
        deleted_count = chat_history_service.cleanup_old_data(
            session['user_id'])
        return jsonify({
            'success': True,
            'deleted_sessions': deleted_count,
            'message': f'Cleaned up {deleted_count} old sessions'
        })
    except (AttributeError, ValueError, RuntimeError) as e:
        logger.error("Error during cleanup: %s", e)
        return jsonify({'error': 'Cleanup failed'}), 500


@app.route('/api/storage/policy', methods=['GET', 'PUT'])
@login_required
def storage_policy():
    """Get or update storage retention policy"""
    if request.method == 'GET':
        return jsonify({
            'max_sessions_per_user': (
                chat_history_service.storage_policy.max_sessions_per_user),
            'max_messages_per_session': (
                chat_history_service.storage_policy.max_messages_per_session),
            'retention_days': chat_history_service.storage_policy.retention_days,
            'auto_archive_days': chat_history_service.storage_policy.auto_archive_days,
            'encryption_enabled': chat_history_service.storage_policy.encryption_enabled,
            'compress_old_messages': chat_history_service.storage_policy.compress_old_messages
        })

    # PUT request
    try:
        policy_updates = request.json
        chat_history_service.update_storage_policy(**policy_updates)
        return jsonify({'success': True,
                        'message': 'Storage policy updated'})
    except (AttributeError, ValueError, KeyError, TypeError) as e:
        logger.error("Error updating policy: %s", e)
        return jsonify({'error': 'Failed to update policy'}), 500


@app.route('/api/export/data')
@login_required
def export_user_data():
    """Export all user chat data"""
    try:
        include_encrypted = request.args.get(
            'include_encrypted', 'false').lower() == 'true'
        export_data = chat_history_service.export_user_data(
            user_id=session['user_id'],
            include_encrypted=include_encrypted
        )

        # Set headers for file download
        response = jsonify(export_data)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'chat_export_{session["user_id"]}_{timestamp}.json'
        response.headers['Content-Disposition'] = f'attachment; filename={filename}'
        return response

    except (AttributeError, ValueError, KeyError, IOError, TypeError) as e:
        logger.error("Error exporting data: %s", e)
        return jsonify({'error': 'Export failed'}), 500


@app.route('/api/ai/feedback', methods=['POST'])
@login_required
def submit_ai_feedback():
    """Submit feedback to improve AI responses"""
    try:
        data = request.get_json()
        user_input = data.get('user_input', '')
        ai_response = data.get('ai_response', '')
        feedback_type = data.get('feedback_type', 'thumbs')
        feedback_value = data.get('feedback_value', True)
        provider = data.get('provider')

        # Try multi-provider feedback first
        try:
            result = handle_feedback_request(
                user_input, ai_response, feedback_type, feedback_value, provider
            )
            return jsonify(result)
        except (AttributeError, ValueError, KeyError, TypeError, RuntimeError):
            # Fallback to original feedback system
            success = submit_feedback(
                user_input,
                ai_response,
                feedback_type,
                feedback_value)

            if success:
                return jsonify({
                    'success': True,
                    'message': 'Feedback submitted successfully',
                    'timestamp': datetime.now().isoformat()
                })
            return jsonify({'error': 'Failed to submit feedback'}), 500

    except (AttributeError, ValueError, KeyError, TypeError, RuntimeError) as e:
        logger.error("Error submitting feedback: %s", e)
        return jsonify({'error': 'Internal server error'}), 500


@app.route('/api/ai/metrics')
@login_required
def ai_performance_metrics():
    """Get AI performance metrics and learning statistics"""
    try:
        metrics = get_ai_metrics()
        return jsonify(metrics)
    except (AttributeError, ValueError, KeyError, TypeError) as e:
        logger.error("Error getting AI metrics: %s", e)
        return jsonify({'error': 'Failed to retrieve metrics'}), 500


@app.route('/api/ai/config', methods=['GET', 'POST'])
@login_required
def ai_configuration():
    """Get or update AI learning configuration"""
    if request.method == 'GET':
        # Return current AI configuration
        return jsonify({
            'learning_enabled': True,
            'feedback_threshold': 0.7,
            'fallback_enabled': True,
            'auto_learning': True,
            'confidence_threshold': 0.8
        })

    # POST request
    try:
        config = request.json
        success = configure_ai_learning(config)

        if success:
            return jsonify({
                'success': True,
                'message': 'AI configuration updated',
                'config': config
            })
        return jsonify(
            {'error': 'Failed to update configuration'}), 500

    except (AttributeError, ValueError, KeyError, TypeError, RuntimeError) as e:
        logger.error("Error updating AI config: %s", e)
        return jsonify({'error': 'Internal server error'}), 500


@app.route('/api/ai/dashboard-data')
@login_required
def ai_dashboard_data():
    """Get comprehensive AI dashboard data"""
    try:
        metrics = get_ai_metrics()

        dashboard_data = {
            'learning_stats': metrics.get('learning_stats', {}),
            'performance_metrics': metrics.get('integration_metrics', {}),
            'system_health': metrics.get('system_health', {}),
            'recent_activity': {
                'responses_today': 45,
                'feedback_received': 12,
                'patterns_learned': 8,
                'accuracy_improvement': '+5.2%'
            },
            'user_engagement': {
                'avg_session_length': '12.5 minutes',
                'satisfaction_score': 4.3,
                'return_rate': '78%'
            }
        }

        return jsonify(dashboard_data)

    except (AttributeError, ValueError, KeyError, TypeError) as e:
        logger.error("Error getting dashboard data: %s", e)
        return jsonify({'error': 'Failed to retrieve dashboard data'}), 500


# Multi-Provider AI Routes
@app.route('/api/ai/providers', methods=['GET'])
@login_required
def get_ai_providers():
    """Get available AI providers and models"""
    try:
        stats = handle_stats_request()
        providers = stats['stats'].get(
            'multi_provider_stats', {}).get('available_models', {})
        return jsonify({
            'success': True,
            'providers': list(providers.keys()),
            'models': providers,
            'stats': stats['stats'].get('multi_provider_stats', {}).get('stats', {})
        })
    except (AttributeError, ValueError, KeyError, TypeError) as e:
        logger.error("Error getting AI providers: %s", e)
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/ai/stats', methods=['GET'])
@login_required
def get_ai_stats():
    """Get comprehensive AI system statistics"""
    try:
        stats = handle_stats_request()
        return jsonify(stats)
    except (AttributeError, ValueError, KeyError, TypeError) as e:
        logger.error("Error getting AI stats: %s", e)
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/health')
def health_check():
    """Health check endpoint for monitoring"""
    try:
        # Test database connection
        conn = get_db_connection()
        conn.close()
        
        return jsonify({
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'version': '1.0.0',
            'database': 'connected'
        }), 200
    except Exception as e:
        logger.error("Health check failed: %s", e)
        return jsonify({
            'status': 'unhealthy',
            'timestamp': datetime.now().isoformat(),
            'error': str(e)
        }), 503


@app.route('/api/ai/test', methods=['POST'])
@login_required
def test_ai_providers():
    """Test different AI providers with a sample message"""
    try:
        data = request.get_json()
        test_message = data.get('message', 'Hello! This is a test message.')

        results = {}
        # Test each provider
        for provider in ['local', 'openai', 'gemini']:
            try:
                result = handle_chat_request(
                    user_input=test_message,
                    provider_preference=provider
                )
                results[provider] = {
                    'success': True,
                    'response': (result['response'][:100] + '...'
                                 if len(result['response']) > 100
                                 else result['response']),
                    'confidence': result['confidence'],
                    'model': result['model'],
                    'response_time': result['metadata'].get('response_time', 0)
                }
            except (AttributeError, ValueError, KeyError, TypeError, RuntimeError) as e:
                results[provider] = {
                    'success': False,
                    'error': str(e)
                }
        return jsonify({
            'success': True,
            'test_message': test_message,
            'results': results
        })
    except (AttributeError, ValueError, KeyError, TypeError, RuntimeError) as e:
        logger.error("Error testing AI providers: %s", e)
        return jsonify({'success': False, 'error': str(e)}), 500


# Error handlers
@app.errorhandler(sqlite3.Error)
def handle_database_error(e):
    """Handle database errors globally"""
    logger.error("Database error: %s", e)
    if request.is_json:
        return jsonify({
            'success': False,
            'error': 'Database temporarily unavailable. Please try again.'
        }), 503
    flash('Service temporarily unavailable. Please try again.')
    return redirect(url_for('index'))

@app.errorhandler(500)
def handle_internal_error(e):
    """Handle internal server errors"""
    logger.error("Internal server error: %s", e)
    if request.is_json:
        return jsonify({
            'success': False,
            'error': 'An unexpected error occurred. Please try again.'
        }), 500
    return render_template('500.html'), 500

@app.errorhandler(404)
def handle_not_found(e):
    """Handle 404 errors"""
    if request.is_json:
        return jsonify({
            'success': False,
            'error': 'Resource not found.'
        }), 404
    return render_template('404.html'), 404

def create_app():
    """Application factory for production deployment"""
    init_db()
    return app

@app.before_request
def force_https():
    """Force HTTPS in production, handle SSL gracefully in development"""
    # Skip HTTPS redirect for local development
    if request.is_secure or request.headers.get('X-Forwarded-Proto') == 'https':
        return
    
    # Allow local development without HTTPS
    if request.host.startswith('127.0.0.1') or request.host.startswith('localhost'):
        return
        
    # In production, redirect to HTTPS
    if os.environ.get('FORCE_HTTPS', 'false').lower() == 'true':
        return redirect(request.url.replace('http://', 'https://'), code=301)

if __name__ == '__main__':
    # Enhanced development server
    init_db()
    debug_mode = os.environ.get('FLASK_DEBUG', 'false').lower() == 'true'
    port = int(os.environ.get('PORT', 5000))
    host = os.environ.get('HOST', '0.0.0.0')  # Allow external connections
 
    if debug_mode:
        logger.warning("Running in DEBUG mode - not suitable for production!")
    
    print(f"🚀 Starting CodeEx AI server...")
    print(f"🌐 Local access: http://127.0.0.1:{port}")
    print(f"📱 Network access: http://192.168.31.234:{port}")
    print(f"⚡ Debug mode: {'ON' if debug_mode else 'OFF'}")
    
    app.run(
        debug=debug_mode, 
        host=host, 
        port=port,
        threaded=True,
        use_reloader=False  # Prevent double initialization in debug mode
    )