#!/usr/bin/env python3
"""
CodeEx AI - Startup Script
Simple script to run the Python web application
"""

import os
import sys
from app import app, init_db


def main():
    """Main startup function"""
    print("🐍 Starting CodeEx AI Python Application...")
    print("=" * 50)

    # Initialize database
    print("📊 Initializing database...")
    init_db()
    print("✅ Database ready!")

    # Check for AI service configuration
    openai_key = os.environ.get('OPENAI_API_KEY')
    anthropic_key = os.environ.get('ANTHROPIC_API_KEY')

    if openai_key:
        print("🤖 OpenAI integration: ENABLED")
    elif anthropic_key:
        print("🤖 Anthropic integration: ENABLED")
    else:
        print("🤖 AI integration: FALLBACK MODE (add API keys for full AI features)")

    # Get configuration
    host = os.environ.get('HOST', '0.0.0.0')
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'

    print(f"🌐 Server starting on http://{host}:{port}")
    print(f"🔧 Debug mode: {'ON' if debug else 'OFF'}")
    print("=" * 50)
    print("🚀 CodeEx AI is ready! Open your browser and start chatting!")
    print("   Press Ctrl+C to stop the server")
    print("=" * 50)

    try:
        app.run(host=host, port=port, debug=debug)
    except KeyboardInterrupt:
        print("\n👋 Shutting down CodeEx AI. Thanks for using our app!")
        sys.exit(0)


if __name__ == '__main__':
    main()
