"""
Authentication and User Management Routes
Separated from main app.py to reduce file size
"""

import sqlite3
import logging
from functools import wraps
from flask import request, jsonify, session, redirect, url_for, flash, render_template
from werkzeug.security import generate_password_hash, check_password_hash
from google.auth.transport import requests as google_requests
from google.oauth2 import id_token
import requests

logger = logging.getLogger(__name__)

# Configuration (will be imported from main app)
GOOGLE_CLIENT_ID = None
GOOGLE_CLIENT_SECRET = None


def init_auth_config(google_client_id, google_client_secret):
    """Initialize authentication configuration"""
    global GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET
    GOOGLE_CLIENT_ID = google_client_id
    GOOGLE_CLIENT_SECRET = google_client_secret


def login_required(f):
    """Decorator to require user login for protected routes"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function


def handle_login():
    """Handle login page logic"""
    if request.method == 'POST':
        data = request.get_json() if request.is_json else request.form
        email = data.get('email')
        password = data.get('password')
        action = data.get('action', 'signin')

        conn = sqlite3.connect('codeex.db')
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

                conn.close()
                if request.is_json:
                    return jsonify({'success': True, 'redirect': '/chat'})
                return redirect(url_for('chat'))

            except sqlite3.IntegrityError:
                conn.close()
                if request.is_json:
                    return jsonify(
                        {'success': False, 'error': 'Email already exists'})
                flash('Email already exists')
                return render_template('login.html', google_client_id=GOOGLE_CLIENT_ID)

        # Sign in
        cursor.execute(
            'SELECT id, password_hash, display_name FROM users WHERE email = ?', (email,))
        user = cursor.fetchone()

        if user and check_password_hash(user[1], password):
            session['user_id'] = user[0]
            session['email'] = email
            session['display_name'] = user[2]

            conn.close()
            if request.is_json:
                return jsonify({'success': True, 'redirect': '/chat'})
            return redirect(url_for('chat'))
        
        conn.close()
        if request.is_json:
            return jsonify(
                {'success': False, 'error': 'Invalid credentials'})
        flash('Invalid credentials')

    return render_template('login.html', google_client_id=GOOGLE_CLIENT_ID)


def handle_google_auth():
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
            conn = sqlite3.connect('codeex.db')
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

            # Set session
            session['user_id'] = user_id
            session['email'] = email
            session['display_name'] = display_name
            session['auth_method'] = 'google'
            session['profile_picture'] = picture

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


def handle_google_callback():
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