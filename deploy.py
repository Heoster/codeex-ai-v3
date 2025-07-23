#!/usr/bin/env python3
"""
Production deployment script for CodeEx AI
"""

import os
import sys
import subprocess
import sqlite3
from pathlib import Path

def check_environment():
    """Check if all required environment variables are set"""
    required_vars = [
        'SECRET_KEY',
        'GOOGLE_CLIENT_ID', 
        'GOOGLE_CLIENT_SECRET',
        'OPENAI_API_KEY',
        'GOOGLE_GEMINI_API_KEY',
        'EMAIL_USERNAME',
        'EMAIL_PASSWORD'
    ]
    
    missing_vars = []
    for var in required_vars:
        if not os.environ.get(var):
            missing_vars.append(var)
    
    if missing_vars:
        print(f"❌ Missing required environment variables: {', '.join(missing_vars)}")
        print("Please set these in your .env file or environment")
        return False
    
    print("✅ All required environment variables are set")
    return True

def install_dependencies():
    """Install Python dependencies"""
    print("📦 Installing dependencies...")
    try:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'], 
                      check=True, capture_output=True)
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def setup_database():
    """Initialize the database"""
    print("🗄️ Setting up database...")
    try:
        from app import init_db
        init_db()
        print("✅ Database initialized successfully")
        return True
    except Exception as e:
        print(f"❌ Failed to initialize database: {e}")
        return False

def check_security():
    """Check security configuration"""
    print("🔒 Checking security configuration...")
    
    # Check if debug mode is disabled
    if os.environ.get('FLASK_DEBUG', 'false').lower() == 'true':
        print("⚠️  WARNING: Debug mode is enabled. Disable for production!")
        return False
    
    # Check if HTTPS is enforced
    if os.environ.get('FORCE_HTTPS', 'false').lower() != 'true':
        print("⚠️  WARNING: HTTPS is not enforced. Enable for production!")
    
    # Check secret key strength
    secret_key = os.environ.get('SECRET_KEY', '')
    if len(secret_key) < 32:
        print("⚠️  WARNING: Secret key should be at least 32 characters long!")
        return False
    
    print("✅ Security configuration looks good")
    return True

def create_systemd_service():
    """Create systemd service file for production deployment"""
    service_content = f"""[Unit]
Description=CodeEx AI Web Application
After=network.target

[Service]
Type=simple
User=www-data
WorkingDirectory={os.getcwd()}
Environment=PATH={os.getcwd()}/venv/bin
ExecStart={os.getcwd()}/venv/bin/gunicorn --bind 0.0.0.0:5000 --workers 4 app:app
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
"""
    
    service_file = Path('/etc/systemd/system/codeex-ai.service')
    try:
        with open(service_file, 'w') as f:
            f.write(service_content)
        print(f"✅ Systemd service file created at {service_file}")
        print("Run 'sudo systemctl enable codeex-ai' to enable auto-start")
        return True
    except PermissionError:
        print("⚠️  Could not create systemd service file (requires sudo)")
        print("Service file content saved to codeex-ai.service")
        with open('codeex-ai.service', 'w') as f:
            f.write(service_content)
        return False

def create_nginx_config():
    """Create nginx configuration"""
    nginx_config = """server {
    listen 80;
    server_name your-domain.com www.your-domain.com;
    
    # Redirect HTTP to HTTPS
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name your-domain.com www.your-domain.com;
    
    # SSL Configuration (update paths to your certificates)
    ssl_certificate /path/to/your/certificate.crt;
    ssl_certificate_key /path/to/your/private.key;
    
    # Security headers
    add_header X-Frame-Options DENY;
    add_header X-Content-Type-Options nosniff;
    add_header X-XSS-Protection "1; mode=block";
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains";
    
    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
    
    # Static files
    location /static {
        alias /path/to/your/app/static;
        expires 1y;
        add_header Cache-Control "public, immutable";
    }
}
"""
    
    with open('nginx-codeex-ai.conf', 'w') as f:
        f.write(nginx_config)
    print("✅ Nginx configuration saved to nginx-codeex-ai.conf")
    print("Copy this to /etc/nginx/sites-available/ and enable it")

def run_tests():
    """Run basic application tests"""
    print("🧪 Running basic tests...")
    try:
        # Test database connection
        from app import get_db_connection
        conn = get_db_connection()
        conn.close()
        print("✅ Database connection test passed")
        
        # Test AI integrations
        from ai_brain_integration import get_ai_metrics
        metrics = get_ai_metrics()
        print("✅ AI integration test passed")
        
        return True
    except Exception as e:
        print(f"❌ Tests failed: {e}")
        return False

def main():
    """Main deployment function"""
    print("🚀 Starting CodeEx AI production deployment...")
    
    steps = [
        ("Environment Check", check_environment),
        ("Install Dependencies", install_dependencies),
        ("Setup Database", setup_database),
        ("Security Check", check_security),
        ("Run Tests", run_tests),
    ]
    
    for step_name, step_func in steps:
        print(f"\n📋 {step_name}...")
        if not step_func():
            print(f"❌ Deployment failed at: {step_name}")
            sys.exit(1)
    
    print("\n🎉 Core deployment completed successfully!")
    
    # Optional steps
    print("\n📋 Creating deployment files...")
    create_systemd_service()
    create_nginx_config()
    
    print("\n✅ CodeEx AI is ready for production!")
    print("\nNext steps:")
    print("1. Configure your domain in nginx-codeex-ai.conf")
    print("2. Set up SSL certificates")
    print("3. Copy nginx config to /etc/nginx/sites-available/")
    print("4. Enable and start the systemd service")
    print("5. Test your deployment")
    
    print("\nTo start the application manually:")
    print("gunicorn --bind 0.0.0.0:5000 --workers 4 app:app")

if __name__ == "__main__":
    main()