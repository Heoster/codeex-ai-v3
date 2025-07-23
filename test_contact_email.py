#!/usr/bin/env python3
"""
Test script for contact form email forwarding to the.heoster@mail.com
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def test_email_configuration():
    """Test email configuration settings"""
    print("🧪 Testing Email Configuration for Contact Form")
    print("=" * 50)

    # Check environment variables
    email_host = os.getenv('EMAIL_HOST')
    email_port = os.getenv('EMAIL_PORT')
    email_username = os.getenv('EMAIL_USERNAME')
    email_password = os.getenv('EMAIL_PASSWORD')

    print(f"📧 Email Host: {email_host}")
    print(f"🔌 Email Port: {email_port}")
    print(f"👤 Email Username: {email_username}")
    print(f"🔑 Email Password: {'✅ Set' if email_password else '❌ Not Set'}")
    print(f"📬 Contact Email: the.heoster@mail.com")
    print()

    # Check if configuration is complete
    if email_host and email_port and email_username and email_password:
        print("✅ Email configuration is complete!")
        return True
    else:
        print("⚠️  Email configuration is incomplete.")
        print("📝 To enable email forwarding:")
        print("   1. Update your .env file with email credentials")
        print("   2. Use Gmail App Password for EMAIL_PASSWORD")
        print("   3. Restart the Flask application")
        print()
        print("📋 Without email configuration:")
        print("   - Contact form will still work")
        print("   - Messages will be logged in Flask logs")
        print("   - Users will receive success confirmation")
        return False


def test_email_sending():
    """Test actual email sending functionality"""
    try:
        # Import the email function from app.py
        sys.path.append('.')
        from app import send_contact_email

        print("🚀 Testing email sending functionality...")

        # Test email data
        test_name = "Test User"
        test_email = "test@example.com"
        test_subject = "Test Contact Form"
        test_message = "This is a test message from the contact form test script."

        # Try to send test email
        result = send_contact_email(
            test_name, test_email, test_subject, test_message)

        if result:
            print("✅ Test email sent successfully!")
            print(f"📧 Email should be delivered to: the.heoster@mail.com")
            print("📋 Check the inbox for the test message.")
        else:
            print("❌ Test email failed to send.")
            print("📝 Check Flask logs for error details.")

        return result

    except ImportError as e:
        print(f"❌ Could not import email function: {e}")
        print("📝 Make sure app.py is in the current directory")
        return False
    except Exception as e:
        print(f"❌ Error testing email: {e}")
        return False


def main():
    """Main test function"""
    print("🧪 CodeEx AI Contact Form Email Test")
    print("=" * 40)
    print()

    # Test configuration
    config_ok = test_email_configuration()
    print()

    if config_ok:
        # Test actual sending
        send_ok = test_email_sending()
        print()

        if send_ok:
            print("🎉 All tests passed! Contact form email forwarding is working.")
            print("📧 All contact form messages will be sent to: the.heoster@mail.com")
        else:
            print("⚠️  Email configuration is set but sending failed.")
            print("📝 Check your email credentials and network connection.")
    else:
        print("📝 Email configuration needed for full functionality.")
        print("💡 Contact form will still work and log messages locally.")

    print()
    print("🔗 Test the contact form at: http://localhost:5000/contact")
    print("📧 All messages will be forwarded to: the.heoster@mail.com")


if __name__ == "__main__":
    main()
