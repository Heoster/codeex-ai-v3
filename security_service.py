"""
🔐 CodeEx AI - Advanced Security Service
Comprehensive security implementation with multi-factor authentication,
rate limiting, encryption, and advanced threat protection
"""

import hashlib
import secrets
import time
import json
import sqlite3
import logging
import re
import base64
import hmac
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
import pyotp
import qrcode
from io import BytesIO
import threading
from functools import wraps
from flask import request, session, jsonify, abort
import ipaddress

logger = logging.getLogger(__name__)

@dataclass
class SecurityEvent:
    """Security event for logging and monitoring"""
    event_type: str
    user_id: Optional[int]
    ip_address: str
    user_agent: str
    timestamp: datetime
    details: Dict
    severity: str  # 'low', 'medium', 'high', 'critical'
    blocked: bool = False

@dataclass
class LoginAttempt:
    """Login attempt tracking"""
    ip_address: str
    user_email: str
    timestamp: datetime
    success: bool
    failure_reason: Optional[str] = None

@dataclass
class SecurityPolicy:
    """Security policy configuration"""
    max_login_attempts: int = 5
    lockout_duration: int = 900  # 15 minutes
    password_min_length: int = 12
    password_require_uppercase: bool = True
    password_require_lowercase: bool = True
    password_require_numbers: bool = True
    password_require_symbols: bool = True
    session_timeout: int = 3600  # 1 hour
    require_2fa: bool = True
    max_sessions_per_user: int = 3
    ip_whitelist: List[str] = None
    enable_rate_limiting: bool = True
    enable_geo_blocking: bool = False
    allowed_countries: List[str] = None

class SecurityService:
    """Advanced security service with comprehensive protection"""
    
    def __init__(self, db_path: str = 'codeex.db'):
        self.db_path = db_path
        self.policy = SecurityPolicy()
        self.rate_limiter = RateLimiter()
        self.session_manager = SessionManager()
        self.audit_logger = AuditLogger(db_path)
        self.encryption_service = EncryptionService()
        self._lock = threading.RLock()
        
        # Initialize security tables
        self._initialize_security_tables()
        
        # Load security configuration
        self._load_security_config()
        
        logger.info("🔐 Security Service initialized with advanced protection")
    
    def _initialize_security_tables(self):
        """Initialize security-related database tables"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Enhanced users table with security fields
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS users_security (
                    user_id INTEGER PRIMARY KEY,
                    password_hash TEXT NOT NULL,
                    salt TEXT NOT NULL,
                    totp_secret TEXT,
                    backup_codes TEXT,  -- JSON array of backup codes
                    failed_login_attempts INTEGER DEFAULT 0,
                    locked_until TIMESTAMP,
                    last_login TIMESTAMP,
                    last_password_change TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    password_history TEXT,  -- JSON array of previous password hashes
                    security_questions TEXT,  -- JSON array of security Q&A
                    account_status TEXT DEFAULT 'active',  -- active, locked, suspended
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (id)
                )
            ''')
            
            # Login attempts tracking
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS login_attempts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ip_address TEXT NOT NULL,
                    user_email TEXT,
                    user_id INTEGER,
                    attempt_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    success BOOLEAN NOT NULL,
                    failure_reason TEXT,
                    user_agent TEXT,
                    country_code TEXT,
                    blocked BOOLEAN DEFAULT FALSE
                )
            ''')
            
            # Active sessions tracking
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS active_sessions (
                    session_id TEXT PRIMARY KEY,
                    user_id INTEGER NOT NULL,
                    ip_address TEXT NOT NULL,
                    user_agent TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_activity TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    expires_at TIMESTAMP NOT NULL,
                    is_active BOOLEAN DEFAULT TRUE,
                    device_fingerprint TEXT,
                    FOREIGN KEY (user_id) REFERENCES users (id)
                )
            ''')
            
            # Security events log
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS security_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_type TEXT NOT NULL,
                    user_id INTEGER,
                    ip_address TEXT NOT NULL,
                    user_agent TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    details TEXT,  -- JSON
                    severity TEXT NOT NULL,
                    blocked BOOLEAN DEFAULT FALSE,
                    resolved BOOLEAN DEFAULT FALSE
                )
            ''')
            
            # Rate limiting tracking
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS rate_limits (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    identifier TEXT NOT NULL,  -- IP or user_id
                    endpoint TEXT NOT NULL,
                    request_count INTEGER DEFAULT 1,
                    window_start TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    blocked_until TIMESTAMP,
                    UNIQUE(identifier, endpoint)
                )
            ''')
            
            # Security configuration
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS security_config (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create indexes for performance
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_login_attempts_ip ON login_attempts(ip_address, attempt_time)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_login_attempts_user ON login_attempts(user_email, attempt_time)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_sessions_user ON active_sessions(user_id, is_active)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_security_events_type ON security_events(event_type, timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_rate_limits_identifier ON rate_limits(identifier, endpoint)')
            
            conn.commit()
    
    def _load_security_config(self):
        """Load security configuration from database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT key, value FROM security_config')
                
                config_data = {}
                for row in cursor.fetchall():
                    try:
                        config_data[row[0]] = json.loads(row[1])
                    except json.JSONDecodeError:
                        config_data[row[0]] = row[1]
                
                # Update policy with loaded configuration
                for key, value in config_data.items():
                    if hasattr(self.policy, key):
                        setattr(self.policy, key, value)
                        
        except Exception as e:
            logger.error(f"Failed to load security config: {e}")
    
    def update_security_policy(self, **kwargs):
        """Update security policy configuration"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            for key, value in kwargs.items():
                if hasattr(self.policy, key):
                    setattr(self.policy, key, value)
                    
                    # Save to database
                    cursor.execute('''
                        INSERT OR REPLACE INTO security_config (key, value, updated_at)
                        VALUES (?, ?, CURRENT_TIMESTAMP)
                    ''', (key, json.dumps(value)))
            
            conn.commit()
            logger.info(f"Security policy updated: {kwargs}")
    
    def hash_password(self, password: str, salt: str = None) -> Tuple[str, str]:
        """Enhanced password hashing with PBKDF2"""
        if salt is None:
            salt = secrets.token_hex(32)
        
        # Use PBKDF2 with SHA-256
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt.encode(),
            iterations=100000,  # High iteration count for security
        )
        
        password_hash = base64.b64encode(kdf.derive(password.encode())).decode()
        return password_hash, salt
    
    def verify_password(self, password: str, stored_hash: str, salt: str) -> bool:
        """Verify password against stored hash"""
        try:
            computed_hash, _ = self.hash_password(password, salt)
            return hmac.compare_digest(computed_hash, stored_hash)
        except Exception as e:
            logger.error(f"Password verification error: {e}")
            return False
    
    def validate_password_strength(self, password: str) -> Tuple[bool, List[str]]:
        """Validate password against security policy"""
        errors = []
        
        if len(password) < self.policy.password_min_length:
            errors.append(f"Password must be at least {self.policy.password_min_length} characters long")
        
        if self.policy.password_require_uppercase and not re.search(r'[A-Z]', password):
            errors.append("Password must contain at least one uppercase letter")
        
        if self.policy.password_require_lowercase and not re.search(r'[a-z]', password):
            errors.append("Password must contain at least one lowercase letter")
        
        if self.policy.password_require_numbers and not re.search(r'\d', password):
            errors.append("Password must contain at least one number")
        
        if self.policy.password_require_symbols and not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
            errors.append("Password must contain at least one special character")
        
        # Check against common passwords
        if self._is_common_password(password):
            errors.append("Password is too common, please choose a more unique password")
        
        return len(errors) == 0, errors
    
    def _is_common_password(self, password: str) -> bool:
        """Check if password is in common passwords list"""
        common_passwords = [
            'password', '123456', '123456789', 'qwerty', 'abc123',
            'password123', 'admin', 'letmein', 'welcome', 'monkey',
            'dragon', 'master', 'shadow', 'superman', 'michael'
        ]
        return password.lower() in common_passwords
    
    def setup_2fa(self, user_id: int, user_email: str) -> Tuple[str, str, List[str]]:
        """Set up two-factor authentication for user"""
        # Generate TOTP secret
        totp_secret = pyotp.random_base32()
        
        # Generate backup codes
        backup_codes = [secrets.token_hex(4).upper() for _ in range(10)]
        
        # Store in database
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO users_security 
                (user_id, totp_secret, backup_codes)
                VALUES (?, ?, ?)
                ON CONFLICT(user_id) DO UPDATE SET
                totp_secret = excluded.totp_secret,
                backup_codes = excluded.backup_codes,
                updated_at = CURRENT_TIMESTAMP
            ''', (user_id, totp_secret, json.dumps(backup_codes)))
            
            conn.commit()
        
        # Generate QR code
        totp = pyotp.TOTP(totp_secret)
        provisioning_uri = totp.provisioning_uri(
            name=user_email,
            issuer_name="CodeEx AI"
        )
        
        # Create QR code
        qr = qrcode.QRCode(version=1, box_size=10, border=5)
        qr.add_data(provisioning_uri)
        qr.make(fit=True)
        
        img = qr.make_image(fill_color="black", back_color="white")
        buffer = BytesIO()
        img.save(buffer, format='PNG')
        qr_code_data = base64.b64encode(buffer.getvalue()).decode()
        
        self.audit_logger.log_event(
            SecurityEvent(
                event_type='2fa_setup',
                user_id=user_id,
                ip_address=request.remote_addr if request else 'system',
                user_agent=request.headers.get('User-Agent', '') if request else 'system',
                timestamp=datetime.now(),
                details={'action': '2fa_enabled'},
                severity='medium'
            )
        )
        
        return totp_secret, qr_code_data, backup_codes
    
    def verify_2fa_token(self, user_id: int, token: str) -> bool:
        """Verify 2FA token (TOTP or backup code)"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT totp_secret, backup_codes FROM users_security 
                    WHERE user_id = ?
                ''', (user_id,))
                
                result = cursor.fetchone()
                if not result:
                    return False
                
                totp_secret, backup_codes_json = result
                
                # Try TOTP first
                if totp_secret:
                    totp = pyotp.TOTP(totp_secret)
                    if totp.verify(token, valid_window=1):  # Allow 30-second window
                        return True
                
                # Try backup codes
                if backup_codes_json:
                    backup_codes = json.loads(backup_codes_json)
                    if token.upper() in backup_codes:
                        # Remove used backup code
                        backup_codes.remove(token.upper())
                        cursor.execute('''
                            UPDATE users_security 
                            SET backup_codes = ?, updated_at = CURRENT_TIMESTAMP
                            WHERE user_id = ?
                        ''', (json.dumps(backup_codes), user_id))
                        conn.commit()
                        
                        self.audit_logger.log_event(
                            SecurityEvent(
                                event_type='backup_code_used',
                                user_id=user_id,
                                ip_address=request.remote_addr if request else 'system',
                                user_agent=request.headers.get('User-Agent', '') if request else 'system',
                                timestamp=datetime.now(),
                                details={'remaining_codes': len(backup_codes)},
                                severity='medium'
                            )
                        )
                        
                        return True
                
                return False
                
        except Exception as e:
            logger.error(f"2FA verification error: {e}")
            return False
    
    def check_login_attempts(self, ip_address: str, email: str = None) -> Tuple[bool, int]:
        """Check if IP or user has exceeded login attempts"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Check IP-based attempts
            cursor.execute('''
                SELECT COUNT(*) FROM login_attempts 
                WHERE ip_address = ? AND success = FALSE 
                AND attempt_time > datetime('now', '-15 minutes')
            ''', (ip_address,))
            
            ip_attempts = cursor.fetchone()[0]
            
            # Check email-based attempts if provided
            email_attempts = 0
            if email:
                cursor.execute('''
                    SELECT COUNT(*) FROM login_attempts 
                    WHERE user_email = ? AND success = FALSE 
                    AND attempt_time > datetime('now', '-15 minutes')
                ''', (email,))
                
                email_attempts = cursor.fetchone()[0]
            
            max_attempts = max(ip_attempts, email_attempts)
            is_blocked = max_attempts >= self.policy.max_login_attempts
            
            return is_blocked, max_attempts
    
    def record_login_attempt(self, ip_address: str, email: str, user_id: int = None, 
                           success: bool = True, failure_reason: str = None):
        """Record login attempt for security monitoring"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            user_agent = request.headers.get('User-Agent', '') if request else ''
            
            cursor.execute('''
                INSERT INTO login_attempts 
                (ip_address, user_email, user_id, success, failure_reason, user_agent)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (ip_address, email, user_id, success, failure_reason, user_agent))
            
            conn.commit()
            
            # Log security event
            self.audit_logger.log_event(
                SecurityEvent(
                    event_type='login_attempt',
                    user_id=user_id,
                    ip_address=ip_address,
                    user_agent=user_agent,
                    timestamp=datetime.now(),
                    details={
                        'email': email,
                        'success': success,
                        'failure_reason': failure_reason
                    },
                    severity='low' if success else 'medium',
                    blocked=not success
                )
            )
    
    def create_secure_session(self, user_id: int, ip_address: str) -> str:
        """Create secure session with tracking"""
        session_id = secrets.token_urlsafe(32)
        expires_at = datetime.now() + timedelta(seconds=self.policy.session_timeout)
        
        user_agent = request.headers.get('User-Agent', '') if request else ''
        device_fingerprint = self._generate_device_fingerprint(user_agent, ip_address)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Clean up old sessions for this user
            cursor.execute('''
                UPDATE active_sessions 
                SET is_active = FALSE 
                WHERE user_id = ? AND is_active = TRUE
            ''', (user_id,))
            
            # Create new session
            cursor.execute('''
                INSERT INTO active_sessions 
                (session_id, user_id, ip_address, user_agent, expires_at, device_fingerprint)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (session_id, user_id, ip_address, user_agent, expires_at, device_fingerprint))
            
            conn.commit()
        
        return session_id
    
    def _generate_device_fingerprint(self, user_agent: str, ip_address: str) -> str:
        """Generate device fingerprint for session tracking"""
        fingerprint_data = f"{user_agent}:{ip_address}"
        return hashlib.sha256(fingerprint_data.encode()).hexdigest()
    
    def validate_session(self, session_id: str, ip_address: str) -> Tuple[bool, int]:
        """Validate session and return user_id if valid"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT user_id, ip_address, expires_at, device_fingerprint
                FROM active_sessions 
                WHERE session_id = ? AND is_active = TRUE
            ''', (session_id,))
            
            result = cursor.fetchone()
            if not result:
                return False, None
            
            user_id, session_ip, expires_at, device_fingerprint = result
            
            # Check if session expired
            if datetime.fromisoformat(expires_at) < datetime.now():
                cursor.execute('''
                    UPDATE active_sessions 
                    SET is_active = FALSE 
                    WHERE session_id = ?
                ''', (session_id,))
                conn.commit()
                return False, None
            
            # Verify IP address (optional strict checking)
            if session_ip != ip_address:
                logger.warning(f"Session IP mismatch: {session_ip} vs {ip_address}")
                # Could be strict and invalidate session, but allowing for now
            
            # Update last activity
            cursor.execute('''
                UPDATE active_sessions 
                SET last_activity = CURRENT_TIMESTAMP 
                WHERE session_id = ?
            ''', (session_id,))
            conn.commit()
            
            return True, user_id
    
    def invalidate_session(self, session_id: str):
        """Invalidate a specific session"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE active_sessions 
                SET is_active = FALSE 
                WHERE session_id = ?
            ''', (session_id,))
            conn.commit()
    
    def invalidate_all_user_sessions(self, user_id: int):
        """Invalidate all sessions for a user"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE active_sessions 
                SET is_active = FALSE 
                WHERE user_id = ?
            ''', (user_id,))
            conn.commit()

class RateLimiter:
    """Advanced rate limiting with multiple strategies"""
    
    def __init__(self):
        self.limits = {
            'login': {'requests': 5, 'window': 300},  # 5 attempts per 5 minutes
            'api': {'requests': 100, 'window': 60},   # 100 requests per minute
            'chat': {'requests': 30, 'window': 60},   # 30 messages per minute
            'signup': {'requests': 3, 'window': 3600} # 3 signups per hour
        }
    
    def is_rate_limited(self, identifier: str, endpoint: str) -> Tuple[bool, int]:
        """Check if identifier is rate limited for endpoint"""
        if endpoint not in self.limits:
            return False, 0
        
        limit_config = self.limits[endpoint]
        
        with sqlite3.connect('codeex.db') as conn:
            cursor = conn.cursor()
            
            # Clean up old entries
            cursor.execute('''
                DELETE FROM rate_limits 
                WHERE window_start < datetime('now', '-{} seconds')
            '''.format(limit_config['window']))
            
            # Check current rate
            cursor.execute('''
                SELECT request_count, blocked_until FROM rate_limits 
                WHERE identifier = ? AND endpoint = ?
            ''', (identifier, endpoint))
            
            result = cursor.fetchone()
            
            if result:
                request_count, blocked_until = result
                
                # Check if still blocked
                if blocked_until and datetime.fromisoformat(blocked_until) > datetime.now():
                    return True, request_count
                
                # Check if limit exceeded
                if request_count >= limit_config['requests']:
                    # Block for window duration
                    block_until = datetime.now() + timedelta(seconds=limit_config['window'])
                    cursor.execute('''
                        UPDATE rate_limits 
                        SET blocked_until = ? 
                        WHERE identifier = ? AND endpoint = ?
                    ''', (block_until.isoformat(), identifier, endpoint))
                    conn.commit()
                    return True, request_count
                
                # Increment counter
                cursor.execute('''
                    UPDATE rate_limits 
                    SET request_count = request_count + 1 
                    WHERE identifier = ? AND endpoint = ?
                ''', (identifier, endpoint))
            else:
                # First request in window
                cursor.execute('''
                    INSERT INTO rate_limits (identifier, endpoint, request_count)
                    VALUES (?, ?, 1)
                ''', (identifier, endpoint))
            
            conn.commit()
            return False, result[0] + 1 if result else 1

class SessionManager:
    """Advanced session management with security features"""
    
    def __init__(self):
        self.active_sessions = {}
        self._lock = threading.RLock()
    
    def create_session(self, user_id: int, session_data: dict) -> str:
        """Create secure session with metadata"""
        with self._lock:
            session_id = secrets.token_urlsafe(32)
            
            session_info = {
                'user_id': user_id,
                'created_at': datetime.now(),
                'last_activity': datetime.now(),
                'ip_address': session_data.get('ip_address'),
                'user_agent': session_data.get('user_agent'),
                'csrf_token': secrets.token_urlsafe(32)
            }
            
            self.active_sessions[session_id] = session_info
            return session_id
    
    def validate_session(self, session_id: str, csrf_token: str = None) -> bool:
        """Validate session with optional CSRF protection"""
        with self._lock:
            if session_id not in self.active_sessions:
                return False
            
            session_info = self.active_sessions[session_id]
            
            # Check CSRF token if provided
            if csrf_token and session_info.get('csrf_token') != csrf_token:
                return False
            
            # Update last activity
            session_info['last_activity'] = datetime.now()
            return True

class AuditLogger:
    """Security audit logging system"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
    
    def log_event(self, event: SecurityEvent):
        """Log security event to database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO security_events 
                    (event_type, user_id, ip_address, user_agent, timestamp, details, severity, blocked)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    event.event_type,
                    event.user_id,
                    event.ip_address,
                    event.user_agent,
                    event.timestamp.isoformat(),
                    json.dumps(event.details),
                    event.severity,
                    event.blocked
                ))
                conn.commit()
                
        except Exception as e:
            logger.error(f"Failed to log security event: {e}")

class EncryptionService:
    """Advanced encryption service for sensitive data"""
    
    def __init__(self):
        self.key = self._load_or_generate_key()
        self.cipher = Fernet(self.key)
    
    def _load_or_generate_key(self) -> bytes:
        """Load or generate encryption key"""
        key_file = '.security_key'
        
        try:
            with open(key_file, 'rb') as f:
                return f.read()
        except FileNotFoundError:
            key = Fernet.generate_key()
            with open(key_file, 'wb') as f:
                f.write(key)
            return key
    
    def encrypt(self, data: str) -> str:
        """Encrypt sensitive data"""
        return self.cipher.encrypt(data.encode()).decode()
    
    def decrypt(self, encrypted_data: str) -> str:
        """Decrypt sensitive data"""
        return self.cipher.decrypt(encrypted_data.encode()).decode()

# Security decorators
def require_auth(f):
    """Decorator to require authentication"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return jsonify({'error': 'Authentication required'}), 401
        return f(*args, **kwargs)
    return decorated_function

def require_2fa(f):
    """Decorator to require 2FA verification"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get('2fa_verified'):
            return jsonify({'error': '2FA verification required'}), 403
        return f(*args, **kwargs)
    return decorated_function

def rate_limit(endpoint: str):
    """Decorator for rate limiting"""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            rate_limiter = RateLimiter()
            identifier = request.remote_addr
            
            is_limited, count = rate_limiter.is_rate_limited(identifier, endpoint)
            if is_limited:
                return jsonify({
                    'error': 'Rate limit exceeded',
                    'retry_after': 300
                }), 429
            
            return f(*args, **kwargs)
        return decorated_function
    return decorator

# Global security service instance
security_service = SecurityService()